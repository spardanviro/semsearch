# semsearch 设计文档

> 基于 EmbeddingGemma 2 的本地语义代码检索，作为 Claude Code 中 grep/ripgrep 的补充。
> 状态：v0.2（2026-11-02，已根据阶段 0 实测结果修订，见 `docs/PHASE0.md`）

---

## 1. 目标与非目标

### 目标
- 让 Claude Code 能用**意图/概念级**查询定位项目文件（"哪里处理限流重试"、"和这段逻辑类似的实现"）。
- 两种调用方式：主会话直接调用 MCP 工具（快）；通过 `semantic-search` subagent（多轮检索+核实，只回传结论）。
- **增量索引**：只重新嵌入发生变化的块，不需要每次全量重建。
- 一套工具服务所有项目，索引存放在全局缓存。
- 数据结构与接口**预留多模态**（图片、PDF、音频、视频），分阶段实现。

### 非目标
- 不替代 Grep/Glob：精确标识符、字符串、正则、"列出全部调用点"仍走原生工具。
- 不做调用图/结构分析：跳转定义、查找引用、调用层级交给 Claude Code 的 LSP 插件（pyright-lsp、csharp-lsp 已安装）。CodeGraph 已弃用并卸载。
- 不把索引或内容发送到任何远程服务，全部在本机完成。

---

## 2. 总体架构

```
┌──────────────────────── Claude Code 会话（可多个） ────────────────────────┐
│ 主会话                                                                      │
│  ├─ Grep / Glob / Read ................ 精确查找（不变）                    │
│  ├─ mcp__semsearch__semantic_search ... 直接语义检索                         │
│  └─ Agent(semantic-search, Haiku) ..... 多轮检索 + Read 核实 → 精炼结论      │
│                                                                              │
│ semsearch-mcp（stdio，每会话一个，轻量，无模型）                              │
│ PostToolUse hook（Edit/Write 后通知变更文件）                                 │
└───────────────┬───────────────────────────────┬─────────────────────────────┘
                │ HTTP 127.0.0.1 + token         │
                ▼                                ▼
┌──────────────────────── semsearchd（全局唯一后台服务） ─────────────────────┐
│ Embedder（sentence-transformers，模型只加载一份，编码器按需加载）            │
│ ProjectRegistry → 每个项目一个 Index（单写者队列）                            │
│   Scanner（rg --files + stat）→ Chunker（tree-sitter / markdown / 行窗口）    │
│   → 块哈希比对 → 只嵌入新块 → Store（SQLite：元数据+向量 BLOB+FTS5）          │
│   VectorCache（内存 numpy 矩阵，KNN 用矩阵乘法，随增量更新同步）              │
│ 空闲 N 分钟自动退出，释放显存                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
        索引位置：~/.cache/semsearch/<项目路径哈希>/index.db
```

### 为什么要单独的后台服务
- MCP stdio 服务每个会话都会起一个实例；模型放在 MCP 里会在 6GB 显存上加载多份。
- 多进程同时写同一个 SQLite 容易锁冲突；后台服务对每个项目只有一个写者。
- hook 脚本是短命进程，需要一个常驻对象来接收"文件已改"的通知。

---

## 3. 组件设计

### 3.1 semsearchd（后台服务）
- 技术：Python 3.12 + FastAPI + uvicorn，只绑定 `127.0.0.1`，随机端口。
- 发现与鉴权：启动时写 `~/.cache/semsearch/daemon.json`：`{pid, port, token, started_at}`；所有请求带 `Authorization: Bearer <token>`，防止本机其他进程随意调用。
- 启动方式：MCP 服务或 hook 发现 daemon 不在时自动拉起（文件锁防止重复启动）。
- 空闲退出：默认 20 分钟无请求即退出（可配置）。
- 接口：

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/search` | 语义检索（见 3.6） |
| POST | `/notify` | 告知变更文件列表，加入更新队列，立即返回 |
| POST | `/reindex` | 指定路径或全量重建 |
| GET | `/status` | 索引状态：文件数、块数、待处理队列、上次更新时间、模型信息 |
| GET | `/health` | 存活检查 |

### 3.2 Embedder（推理后端）
- 库：`sentence-transformers >= 6.1`，torch CUDA 版，bf16。
- 模型：`google/embeddinggemma-2`。
- 接口（后端可替换）：

```python
class Embedder(Protocol):
    model_id: str
    dim: int  # 768
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...       # prompt_name="SearchQuery"
    def encode_documents(self, items: list[Item]) -> np.ndarray: ...    # 文本 prompt_name="Document"；媒体不加 prompt
```

- `Item` 是带模态的输入：`TextItem(text)`、`ImageItem(path|PIL)`、`AudioItem(path)`、`VideoItem(path)`、或交错输入。
- **编码器按需加载**：
  - 默认只加载文本编码器（通过 `config_kwargs` 把 `vision_config` / `audio_config` 设为 `None`）。
  - 遇到媒体时切换为带对应编码器的实例；切换后旧实例释放。
  - 阶段 0 实测：两种加载方式对同一文本的向量余弦最小 0.996，**不完全一致**。因此 `meta` 记录 `load_mode`（text / full），加载方式变化即全量重建，不混用。
- 前缀（阶段 0 实测选定）：查询 `SearchQuery` = `task: search result | query: `；文档 `Document` = `title: none | text: `。`CodeRetrieval` 实测更差，不使用。
- 批处理：按 token 预算组 batch（目标峰值显存 ≤ 2.5GB，约等于 16 × 1024 tokens），OOM 时减半重试。实测 3060 上约 69 块/秒。
- 输出先转 float32 再 L2 归一化，存 768 维（128 维会明显降低中文查询效果）。
- 依赖注意：即使只加载文本部分，也必须安装 Pillow 和 torchvision。

### 3.3 Scanner（文件发现）
- 首选 `rg --files --hidden`（自动遵守 .gitignore/.ignore），找不到 rg 时退回 `pathspec` 解析 .gitignore。
- 额外忽略：`.git/`、`node_modules/`、构建产物、锁文件，以及项目根的 `.semsearchignore`。
- **默认不索引敏感文件**：`.env*`、`*.pem`、`*.key`、`id_rsa*`、`*.pfx`、`secrets.*` 等（返回给 Claude 的片段会进入上下文）。
- 文本文件大小上限 1MB；二进制检测（前 8KB 含 NUL）后跳过，除非属于已启用的媒体类型。
- 模态判定按扩展名：代码/文本 → `code`/`text`；`.png/.jpg/.webp` → `image`；`.pdf` → `pdf`；`.wav/.mp3/.flac` → `audio`；`.mp4/.mov/.webm` → `video`。未启用的模态直接跳过。

### 3.4 Chunker（分块）
每个块包含：`modality, locator, symbol, kind, text_for_embedding, preview`。

| 内容 | 策略 |
|---|---|
| 代码（tree-sitter 支持的语言） | 按顶层函数/类/方法切；类过大则按方法拆；顶部 import/常量区单独一块 |
| 代码（不支持的语言） | 行窗口：60 行，重叠 15 行，尽量在空行处断开 |
| Markdown / RST | 按标题层级切，块内保留标题路径 |
| 其他文本（json/yaml/toml 等） | 行窗口；小文件整文件一块 |
| PDF（阶段 2） | 每页一块：提取文字 + 渲染页面图像，交错输入 |
| 图片（阶段 2） | 每张一块，附文件名作为文本上下文 |
| 音频/视频（阶段 3） | 按时间窗口切段（如 30s，重叠 5s），locator 记录起止时间 |

- 嵌入文本加头部，提升检索质量：

```
path: src/net/retry.py
symbol: RetryPolicy.backoff
kind: method
lang: python
---
<代码正文>
```

- 块长度用模型 tokenizer 计数，目标 200–800 tokens，硬上限 1024；超过则按子节点（类→方法）或行窗口二次拆分。阶段 0 实测约 2% 的定义超过 1024 tokens，最长 3157。
- tree-sitter 依赖：`tree-sitter-language-pack`，Windows 上已验证 Python / TypeScript / Go / C#（101KB 的 C# 文件解析 21ms）。

### 3.5 Store（存储）
SQLite（WAL 模式）+ FTS5，一个项目一个 `index.db`。向量以 BLOB 存在 `chunks.embedding` 中，作为唯一数据源。

> 阶段 0 实测：sqlite-vec 的 vec0 在 5 万条上 KNN 要 93ms，numpy 暴力计算只要 3ms，写入和删除也慢得多，因此不使用 sqlite-vec。

**VectorCache**：后台服务为每个已打开的项目在内存中维护一个 `float32[N, 768]` 矩阵和 `chunk_id` 数组。
- 启动或首次访问项目时从 SQLite 读入。
- 增量更新提交事务后同步修改：删除用标记 + 定期压缩；新增追加在末尾。
- 10 万个块约占 300MB 内存；内存紧张时可改存 float16，减半。

```sql
CREATE TABLE meta (
  key TEXT PRIMARY KEY, value TEXT
);  -- schema_version, model_id, dim, chunker_version, project_root

CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,          -- 相对项目根，统一用 '/'
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  content_hash TEXT NOT NULL,         -- blake3
  modality TEXT NOT NULL,
  indexed_at INTEGER NOT NULL
);

CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_hash TEXT NOT NULL,           -- blake3(text_for_embedding 或媒体字节+locator)
  modality TEXT NOT NULL,
  start_line INTEGER, end_line INTEGER,   -- 文本/代码
  page INTEGER,                           -- PDF
  t_start REAL, t_end REAL,               -- 音视频（秒）
  symbol TEXT, kind TEXT,
  preview TEXT NOT NULL,                  -- 返回给 Claude 的短片段（<= 600 字符）
  embedding BLOB NOT NULL                 -- float32[768]，已 L2 归一化
);
CREATE INDEX idx_chunks_file ON chunks(file_id);
CREATE INDEX idx_chunks_hash ON chunks(chunk_hash);

-- 阶段 1.5：混合检索
CREATE VIRTUAL TABLE fts_chunks USING fts5(
  text, symbol, path, content='', tokenize='unicode61'
);
```

- 嵌入复用：同一 `chunk_hash` 已存在向量时直接复制，不重新计算（文件重命名、代码搬移时尤其有用）。
- `meta` 中 `model_id / dim / load_mode / chunker_version` 任一不一致 → 自动全量重建。

### 3.6 检索流程
1. 处理检索前先做**新鲜度检查**（见第 4 节），把脏文件同步更新完，或在超时预算内尽量更新。
2. `encode_queries([query])` → 与 VectorCache 做矩阵乘法 → `argpartition` 取 `k * 4` 个候选。
3. 过滤：`path_glob`、`modality`、`min_score`。
4. 去重：每个文件最多保留 2 个块；相邻/重叠块合并为一个行区间。
5. （阶段 1.5）混合检索：FTS5 BM25 结果与向量结果做 RRF 融合，对查询里包含标识符的场景更稳。
6. 返回 top-k：

```json
{
  "results": [
    {
      "path": "src/net/retry.py",
      "lines": [42, 88],
      "symbol": "RetryPolicy.backoff",
      "modality": "code",
      "score": 0.71,
      "preview": "def backoff(self, attempt): ..."
    }
  ],
  "index": {"fresh": true, "pending": 0, "files": 1834, "chunks": 12410}
}
```

---

## 4. 增量更新

### 4.1 变更检测
| 触发 | 时机 | 覆盖场景 |
|---|---|---|
| 检索前快速扫描 | 每次 `/search`，同一项目 5 秒内不重复扫 | git checkout、外部编辑器、其他工具改文件 |
| PostToolUse hook | Claude 每次 Edit/Write/MultiEdit/NotebookEdit 之后 | Claude 自己的修改，最及时 |
| 文件监听（可选，后续） | watchdog 实时事件 | 长时间运行的大仓库 |

### 4.2 更新算法（单个项目）
```
files_now  = scanner.list()                      # rg --files
stat 比较 files 表：
  新文件 / size 或 mtime 变化 → 读内容算 content_hash
      hash 未变 → 只更新 mtime（例如 touch）
      hash 变化 → 重新分块：
          新块 hash 已存在于库 → 复用向量
          否则 → 加入嵌入批次
          旧块中不再出现的 → 删除（chunks + vec + fts）
  files 表有但磁盘已无 → 删除文件及其块
批量嵌入 → 单个事务提交
```
- 编辑一个函数，通常只重新嵌入 1–2 个块。
- 首次索引在后台进行，期间检索返回已完成部分，并在 `index.fresh=false` 中标明进度。
- 每个项目一个写者协程；`/notify` 只入队，不阻塞 hook。

### 4.3 新鲜度预算
- `/search` 默认最多等待 1.5 秒完成待更新项，超时则用现有索引作答并标注 `pending`。

---

## 5. Claude Code 集成

### 5.1 MCP 服务（semsearch-mcp）
- 用官方 `mcp` Python SDK，stdio 传输。
- 项目根：优先 `CLAUDE_PROJECT_DIR` 环境变量，其次进程 cwd；向上查找 `.git` 作为根。
- 工具：

| 工具 | 参数 | 说明 |
|---|---|---|
| `semantic_search` | `query`, `k=8`, `path_glob?`, `modality?`, `min_score?` | 主要入口 |
| `index_status` | — | 查看索引进度与健康状态 |
| `reindex` | `paths?`, `full=false` | 手动触发重建 |

- `semantic_search` 的工具描述直接写入路由规则（见 5.4），让主会话自己判断何时该用。

### 5.2 PostToolUse hook
`~/.claude/settings.json`：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit|NotebookEdit",
        "hooks": [{ "type": "command", "command": "semsearch-hook" }]
      }
    ]
  }
}
```
- 从 stdin 读取 hook JSON，取出 `tool_input.file_path`，POST 到 `/notify`。
- 超时 300ms；失败不影响 Claude 的操作，但写入 `~/.cache/semsearch/hook.log`（不静默吞错误）。
- daemon 未运行时什么也不做（只记日志），不为一次 hook 拉起模型；下次检索前的快速扫描会补上这次改动。
- daemon 只更新**已经建过索引**的项目：在从没用过语义检索的项目里编辑文件，不会触发整个项目的首次索引。

### 5.3 Subagent：`~/.claude/agents/semantic-search.md`（草稿）

```markdown
---
name: semantic-search
description: 按意图或概念在当前项目中定位代码/文档。适用于不知道确切名称的问题，
  例如"哪里处理鉴权失败重试"、"有没有类似 X 的实现"。已知标识符、字符串或正则时
  不要用它，直接用 Grep。返回已核实的文件位置和一句话说明。
tools: mcp__semsearch__semantic_search, Read, Grep, Glob
model: haiku
---
你是项目内的语义检索助手。

流程：
1. 用 semantic_search 检索原始问题；必要时换 1–2 种说法（同义改写、拆成子问题）再查。
   如果问题是中文，至少再用英文改写查一次（代码和注释大多是英文，阶段 0 实测中文召回较弱）。
2. 对得分靠前的结果用 Read 读取对应行区间，确认确实相关；不相关的丢弃。
3. 发现确切的符号名后，可以用 Grep 补全它的定义与主要引用。
4. 最多 6 次工具调用后结束。

输出（只输出这些）：
- 每条：`path:start-end` — 符号名 — 一句话说明为什么相关
- 最多 8 条，按相关度排序
- 若没有可信结果，明确说"未找到"，并给出建议的 Grep 关键词
```

### 5.4 路由规则（同时写入工具描述和用户级 CLAUDE.md）
| 情况 | 用什么 |
|---|---|
| 已知函数名/类名/变量名、字符串、错误信息原文、正则 | Grep |
| 找文件名/路径模式 | Glob |
| 需要完整列出所有调用点或出现位置 | Grep |
| 跳转定义、查找引用、调用层级、改动影响面 | LSP |
| 不知道名字，只知道功能或意图 | semantic_search |
| "有没有类似这段的代码"、跨文件找同一概念 | semantic_search |
| 开放式探索，需要多轮查找和核实 | semantic-search subagent |
| 语义检索找到符号后，确认全部引用 | 转到 LSP（查找引用）或 Grep |

### 5.5 安装方式
```bash
# 在仓库根目录执行；非 editable 安装，包被复制到 uv 工具目录的独立环境里，
# 之后移动或删除仓库不影响使用。pyproject 的 [tool.uv.sources] 保证 torch 装 cu130 版。
uv tool install . --python 3.12
semsearch install-claude             # 注册 MCP、写入 hook、subagent、CLAUDE.md 路由片段（先展示改动再确认）
```
- 更新代码后：`uv tool install . --python 3.12 --reinstall`，再 `semsearch daemon stop`，让新代码生效。
- 卸载：`semsearch install-claude --uninstall`，然后 `uv tool uninstall semsearch`。
- 写入的配置指向 `%APPDATA%\uv\tools\semsearch\Scripts\` 下的绝对路径，不依赖 PATH。

CLI 也可单独使用：`semsearch search "限流重试"`、`semsearch status`、`semsearch reindex`。

---

## 6. 项目结构

```
SearchEngine/
├─ pyproject.toml
├─ docs/DESIGN.md
├─ src/semsearch/
│  ├─ config.py           # 配置与默认值（端口、超时、忽略规则、块大小）
│  ├─ daemon/
│  │  ├─ app.py           # FastAPI 路由
│  │  ├─ lifecycle.py     # 启动锁、daemon.json、空闲退出
│  │  └─ registry.py      # 项目 → Index 管理、写者队列
│  ├─ embed/
│  │  ├─ base.py          # Embedder 协议、Item 类型
│  │  └─ st_backend.py    # sentence-transformers 实现、编码器按需加载
│  ├─ index/
│  │  ├─ scanner.py
│  │  ├─ chunkers/        # code.py / markdown.py / window.py / pdf.py / media.py
│  │  ├─ store.py         # SQLite schema、迁移、读写
│  │  ├─ updater.py       # 增量更新算法
│  │  └─ search.py        # KNN、过滤、去重、RRF
│  ├─ client.py           # 连接/拉起 daemon 的共享客户端
│  ├─ mcp_server.py
│  ├─ hook.py
│  └─ cli.py
├─ integrations/claude/
│  ├─ agents/semantic-search.md
│  └─ claude-md-snippet.md
├─ eval/
│  ├─ queries/<repo>.jsonl   # {query, expected_paths, kind}
│  └─ run_eval.py
└─ tests/                    # unit / integration / e2e
```

---

## 7. 分阶段计划

### 阶段 0：可行性验证（已完成，见 `docs/PHASE0.md`）
- [x] 用 uv 建环境，安装 CUDA 版 torch（2.14.1+cu130）
- [x] 下载 `google/embeddinggemma-2`（不需要登录）
- [x] 确认 prompt 名称与完整列表
- [x] 验证仅文本加载与全模态加载的一致性（cos 最小 0.996，不混用）
- [x] 测量吞吐与显存（约 69 块/秒，batch 16 时峰值 2.5GB）
- [x] 验证 sqlite-vec（可用但慢，弃用）与 tree-sitter（可用）

**结论**：通过。1 万个块约 2.5 分钟。

### 阶段 1：文本与代码 MVP（已完成）
- [x] Store + schema
- [x] Scanner、代码/Markdown/行窗口分块器
- [x] 增量更新算法与块级向量复用
- [x] semsearchd（search / notify / status / reindex、启动锁、空闲退出）
- [x] MCP 服务、CLI
- [x] PostToolUse hook
- [x] subagent 与路由规则、安装器
- [x] 测试：单元、集成（daemon + SQLite）、端到端（真实子进程 daemon、真实 stdio MCP 调用）

**结果**：102 个测试通过（另有 2 个按条件跳过：真模型 GPU 测试、需要开发者模式的符号链接测试），覆盖率 91%，ruff / pyright 无问题。真实项目（RaftMod-Publish，235 个文件，3,076 块）首次索引约 82 秒，热查询约 0.7 秒（CLI，含进程启动），冷启动约 19 秒。改一个文件只重新嵌入变化的块（实测 1 块）。

实现中新增的行为：
- 内容完全相同的块（复制出来的代码树）合并成一条结果，并注明其他副本路径。
- 返回结果前检查文件是否还存在：shell 里删掉、hook 没通知到的文件不会出现在结果里，并自动安排清理。
- 只有 import / using / 花括号的代码块不索引；空白区块会去掉开头的 import，预览从有意义的内容开始。

代码审查与安全审查后的加固：
- 单个文件分块失败时退回行窗口，再失败就跳过该文件，不影响其他文件；临时读取失败（文件被占用）下次扫描重试。
- 行号只按 `\n` 计算，与 tree-sitter 一致；遍历语法树改为显式栈，深层嵌套不会递归溢出。
- SQLite 提交失败会回滚；chunk id 用 AUTOINCREMENT，不复用；向量缓存在数据库锁内同步更新。
- 类型过滤（code / text）在排序前生效。
- 不跟随符号链接和 junction，不读取项目目录之外的文件；敏感文件规则不能被 `.semsearchignore` 的 `!` 取消；每个项目最多 5 万个文件（`SEMSEARCH_MAX_FILES`）。
- 检索结果中的路径、符号名去掉换行和控制字符，预览加 `|` 前缀，并提示其中内容不可信。
- daemon 校验 Host 头（防 DNS rebinding），token 文件以仅本人可读的方式创建；客户端不走系统代理；只允许一个 daemon 实例；启动锁在成功或失败后都会释放。
- Windows 上调用 rg / git 不再弹出控制台窗口；hook 通知的文件同样遵守 .gitignore；`bin` / `build` 等目录只在项目没有 .gitignore 时才默认排除。

### 阶段 1.5：检索质量
- [ ] FTS5 混合检索 + RRF
- [ ] 评估集与 `run_eval.py`，调块大小、头部格式、去重策略

### 阶段 2：图片与 PDF
- [ ] 视觉编码器按需加载
- [ ] PDF 分页（文字 + 页面图像）、图片分块
- [ ] 结果中返回页码/图片路径

### 阶段 3：音频与视频
- [ ] 音频编码器按需加载、时间窗口切段（16kHz 单声道重采样）
- [ ] 视频按 1fps 抽帧，结果返回时间段

---

## 8. 评估方法
- 选 2–3 个真实仓库（包含你自己的项目和一个开源中型仓库，如 HF transformers 的子目录）。
- 每个仓库写 20–30 条查询，分三类：
  - 概念类："哪里做请求重试"
  - 描述类："把配置从环境变量读进来的地方"
  - 相似类：给一段代码，找类似实现
- 指标：Recall@5、Recall@10、MRR；同时记录"用 grep 需要几次尝试才能找到"作为对照。
- 每次调整分块或检索策略都跑一遍，结果写入 `eval/results/`。

---

## 9. 风险与待确认事项
| 项 | 风险 | 处理 |
|---|---|---|
| 中文查询召回偏弱 | 阶段 0 中文 R@10 为 80%，英文为 95% | subagent 加英文改写；阶段 1.5 加 FTS5 混合检索；用真实项目评估集复核 |
| 大型 C# 项目首次索引慢 | 1 万多个文件可能要 20–30 分钟 | 后台索引、返回部分结果并标注进度；阶段 1 尝试按 token 预算组 batch 提速 |
| 6GB 显存 | 全模态 + 大批量可能 OOM | 自适应批大小；默认只加载文本编码器；空闲释放 |
| Windows 兼容 | 路径分隔符与大小写（依赖兼容性已在阶段 0 验证） | 路径统一规范化为小写盘符 + `/` |
| 大仓库首次索引耗时 | 几十万行代码可能需要数分钟 | 后台索引、检索返回部分结果并标注进度 |
| 敏感信息 | 索引片段会进入 Claude 上下文 | 默认忽略密钥类文件；支持 `.semsearchignore` |
| 本机其他进程访问 daemon | 可读取项目内容 | 仅绑定 127.0.0.1 + token；拒绝项目根之外的路径 |
| 多会话并发 | 同一项目多处同时触发更新 | 每项目单写者队列，SQLite WAL |

---

## 10. 依赖清单（初步）
- 运行时：`torch` + `torchvision`（cu130）、`pillow`、`sentence-transformers>=6.1`、`transformers`、`fastapi`、`uvicorn`、`httpx`、`mcp`、`tree-sitter-language-pack`、`blake3`、`pathspec`、`numpy`
- 阶段 2：`pymupdf`（PDF 文字与渲染）
- 阶段 3：`soundfile` / `librosa`（重采样）、`av` 或 `decord`（视频抽帧）
- 开发：`pytest`、`pytest-cov`、`ruff`、`mypy`
- 外部工具：`ripgrep`（可选但推荐）
