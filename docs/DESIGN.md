# semsearch 设计文档

> 基于 EmbeddingGemma 2 的本地语义代码检索，作为 Claude Code 中 grep/ripgrep 的补充。
> 状态：v0.3（2026-11-03，按全局审校后的实现修订；审校记录见 `docs/AUDIT.md`）

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
- 不做调用图/结构分析：跳转定义、查找引用、调用层级交给 Claude Code 的 LSP 插件（语言服务器按项目语言自行安装）。
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
│   Scanner（git ls-files / rg --files / 目录遍历 + stat）                      │
│   → Chunker（tree-sitter / markdown / 行窗口 / 媒体分段）                     │
│   → 块哈希比对 → 只嵌入新块 → Store（SQLite：元数据+向量 BLOB+FTS5）          │
│   VectorCache（内存 numpy 矩阵，KNN 用矩阵乘法，随增量更新同步）              │
│ 空闲 20 分钟自动退出，释放显存                                                │
└──────────────────────────────────────────────────────────────────────────────┘
        索引位置：~/.cache/semsearch/projects/<项目路径 sha1 前 16 位>/index.db
        （SEMSEARCH_HOME 可改；整个目录只允许当前用户访问）
```

### 为什么要单独的后台服务
- MCP stdio 服务每个会话都会起一个实例；模型放在 MCP 里会在 6GB 显存上加载多份。
- 多进程同时写同一个 SQLite 容易锁冲突；后台服务对每个项目只有一个写者。
- hook 脚本是短命进程，需要一个常驻对象来接收"文件已改"的通知。

---

## 3. 组件设计

### 3.1 semsearchd（后台服务）
- 技术：Python 3.12 + FastAPI + uvicorn，只绑定 `127.0.0.1`，随机端口。
- 发现与鉴权：启动时原子写入 `~/.cache/semsearch/daemon.json`：`{pid, port, token, started_at}`（Windows 上读者占用时重试替换）；所有请求带 `Authorization: Bearer <token>`，用常数时间比较；同时校验 Host 头，防 DNS rebinding。
- 启动方式：MCP 服务和 CLI 发现 daemon 不在时自动拉起（文件锁防止重复启动）；用 `python -I -m semsearch.daemon` 启动，不受当前目录里同名模块影响。启动后轮询 `/health`，要求返回的 pid 与 daemon.json 一致；子进程提前退出时立即报错并附上日志末尾。hook 从不拉起 daemon。
- 空闲退出：默认 20 分钟无请求即退出（`SEMSEARCH_IDLE_TIMEOUT`）。
- 拒绝的项目根：用户主目录、磁盘根、`~/.claude`、`~/.ssh` 和 semsearch 自己的数据目录。
- 接口（全部需要 token）：

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/search` | 语义检索（见 3.6） |
| POST | `/notify` | 告知变更文件列表（最多 1000 个），加入更新队列，立即返回 |
| POST | `/reindex` | 指定路径或全量重建 |
| POST | `/status` | 某个项目的索引状态：文件数、块数、进度、上次扫描与错误、模型信息 |
| POST | `/shutdown` | 让 daemon 退出（`semsearch daemon stop`） |
| GET | `/health` | 存活检查，返回 pid |

- 请求限制（`daemon/limits.py`）：请求体最多 1 MB（413）；最多 32 个并发连接（uvicorn，503）；持有 token 的请求共用一个令牌桶，每秒 20 次、可突发 100 次（429 + Retry-After），`/health` 不限速。客户端遇到 429/503 时按 Retry-After 重试。
- 请求参数由 pydantic 校验（例如 `k` 为 1–50，`modality` 只能是已知类型）；参数错误返回 400/422，客户端报成 `DaemonRequestError`，不会被当成"daemon 不可用"。

### 3.2 Embedder（推理后端）
- 库：`sentence-transformers >= 6.1`，torch CUDA 版，bf16。
- 模型：默认 `google/embeddinggemma-2`，固定在测试过的 Hugging Face 版本（`SEMSEARCH_MODEL_REVISION`）；可通过 `SEMSEARCH_MODEL` 换成任意 sentence-transformers 模型。
- 接口（`embed/base.py`，后端可替换；`SEMSEARCH_EMBEDDER=fake` 是测试用的词袋哈希实现）：

```python
class Embedder(Protocol):
    model_id: str
    load_mode: str                     # text / text+vision / text+audio / full
    @property
    def dim(self) -> int: ...
    def count_tokens(self, text: str) -> int: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...    # 查询提示词
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...  # 文档提示词
    def encode_media(self, items: list[dict]) -> np.ndarray: ...     # {"image"|"audio"|"video": ...}
                                                                     # 或交错 {"text": "... <|image|>", "image": [...]}
```

- **编码器按配置加载**：
  - `SEMSEARCH_MEDIA` 未开启时只加载文本编码器（通过 `config_kwargs` 把 `vision_config` / `audio_config` 设为 `None`）。
  - 开启后按需要的媒体类型加载视觉/音频编码器，并把最大序列长度放宽到模型原生上限（否则图片 token 会被截断）。
  - 阶段 0 实测：不同加载方式对同一文本的向量余弦最小 0.996，**不完全一致**。因此 `meta` 记录 `load_mode`，加载方式变化即全量重建，不混用。
- 前缀（阶段 0 实测选定）：查询 `SearchQuery` = `task: search result | query: `；文档 `Document` = `title: none | text: `。`CodeRetrieval` 实测更差，不使用。
- 批处理：按 token 预算组 batch（目标峰值显存 ≤ 2.5GB，约等于 16 × 1024 tokens），OOM 时减半重试。实测 3060 上约 69 块/秒。
- 输出先转 float32 再 L2 归一化，存 768 维（128 维会明显降低中文查询效果）。
- 依赖注意：即使只加载文本部分，也必须安装 Pillow 和 torchvision。

### 3.3 Scanner（文件发现）
- 列文件的顺序：`git ls-files -co --exclude-standard`（git 仓库）→ `rg --files --hidden --no-require-git` → 自己遍历目录，用 `pathspec` 按 git 的优先级读取每一级的 `.gitignore` 和 `.git/info/exclude`。先用 git 是为了和 hook 的 `git check-ignore` 判断一致，避免已跟踪但被忽略的文件反复增删。
- 额外忽略：`.git/`、`node_modules/`、构建产物、锁文件，以及项目根的 `.semsearchignore`。
- 来自仓库的忽略规则和检索时的 `path_glob` 都是不可信输入：用 RE2 后端匹配（线性时间，防 ReDoS），并限制行数和长度。
- **默认不索引敏感文件**：`.env*`、`*.pem`、`*.key`、`id_rsa*`、`*.pfx`、`secrets.*` 等（返回给 Claude 的片段会进入上下文）。不区分大小写，`!` 规则不能取消。
- 文本文件大小上限 1MB，json/yaml/xml 等配置文件 200KB；二进制检测（前 8KB 含 NUL）后跳过，除非属于已启用的媒体类型。平均每行非空白字符超过 500 的压缩文件跳过。带 UTF-16 BOM 的文本按 UTF-16 解码。
- 不跟随符号链接和 junction；每个项目最多 5 万个文件。
- 模态判定按扩展名：代码/文本 → `code`/`text`；图片、`.pdf`、音频、视频扩展名见 `config.MEDIA_EXTS`。未启用的模态直接跳过。

### 3.4 Chunker（分块）
每个块包含：`modality, locator, symbol, kind, text_for_embedding, preview`。

| 内容 | 策略 |
|---|---|
| 代码（tree-sitter 支持的语言） | 按顶层函数/类/方法切；类过大则按方法拆；紧贴定义的注释和 Rust 属性并入该定义；定义之间的零散代码不少于 80 个字符才单独成块 |
| 代码（不支持的语言） | 行窗口：60 行，重叠 15 行，尽量在空行处断开 |
| Markdown | 按标题层级切，块内保留标题路径；代码围栏里的 `#` 不当标题 |
| RST 及其他文本（json/yaml/toml 等） | 行窗口；小文件整文件一块 |
| 超长单行 | 按 token 上限拆开，避免后半段对嵌入不可见 |
| PDF（阶段 2） | 每页一块：提取文字 + 渲染页面图像，交错输入；没有文字的页只用图像 |
| 图片（阶段 2） | 每张一块，只用像素；小于 32×32 的跳过，超大图片按像素上限拒绝 |
| 音频（阶段 3） | 30 秒窗口，步长 25 秒（重叠 5 秒） |
| 视频（阶段 3） | 20 秒一段，每秒 1 帧；启用音频时同段声音一起编码 |

- tree-sitter 识别的特殊名字：PowerShell 函数、JS 箭头函数赋值、Go/Rust 的方法与 impl。语法树最多下钻 30 层容器，再深就按行窗口处理。
- 媒体文件的分段和解码在子进程 `index/media_worker.py` 里进行：内存上限 2 GB，每次调用限时 120 秒，Windows 上禁止再启动进程。子进程只回传 JSON 和不经 pickle 的 numpy 数组，daemon 检查后再拼成模型输入（`media.assemble`）。子进程在每次增量更新开始时启动、结束时退出。
- 某种语言的解析器加载失败（例如首次下载时离线）时先退回行窗口，300 秒后重试。

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
- 增量更新在事务 COMMIT 成功后才同步修改缓存（提交失败时缓存不变）：删除用标记 + 定期压缩；新增追加在末尾。
- 10 万个块约占 300MB 内存；内存紧张时可改存 float16，减半。
- 需要 SQLite ≥ 3.43（contentless_delete 的 FTS5 表），启动时检查。

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,          -- 相对项目根，统一用 '/'
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  content_hash TEXT NOT NULL,         -- blake3
  modality TEXT NOT NULL,
  lang TEXT NOT NULL,
  indexed_at REAL NOT NULL
);

CREATE TABLE chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,   -- 不复用 id，旧缓存行不会配上别的块
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_hash TEXT NOT NULL,           -- blake3(text_for_embedding 或媒体字节+locator)
  modality TEXT NOT NULL,
  start_line INTEGER, end_line INTEGER,   -- 文本/代码
  page INTEGER,                           -- PDF
  t_start REAL, t_end REAL,               -- 音视频（秒）
  symbol TEXT, kind TEXT,
  preview TEXT NOT NULL,                  -- 返回给 Claude 的短片段（<= 600 字符）
  tokens INTEGER NOT NULL,
  embedding BLOB NOT NULL                 -- float32[dim]，已 L2 归一化
);
CREATE INDEX idx_chunks_file ON chunks(file_id);
CREATE INDEX idx_chunks_hash ON chunks(chunk_hash);

-- 阶段 1.5：关键词索引（rowid = chunks.id；只存词，不存原文）
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  meta, body, content='', contentless_delete=1,
  tokenize="unicode61 remove_diacritics 2 tokenchars '_'"
);
```

- 嵌入复用：同一 `chunk_hash` 已存在向量时直接复制，不重新计算（文件重命名、代码搬移时尤其有用）。
- `meta` 的键：`schema_version`、`chunker_version`、`keywords_version`、`model_id`、`model_revision`、`dim`、`load_mode`、`root`、`document_prompt`、`chunk_tokens`，开启媒体时还有 `media`（类型列表、媒体版本和各项上限）。任一不一致 → 自动全量重建；只改查询提示词不重建。
- 全量重建时先扫描文件、再清空旧索引：扫描失败（例如目录暂时不可读）不会留下一个空索引。扫描失败后 60 秒内不再重试。
- 嵌入失败分两类：临时错误（显存不足、文件被占用）逐条重试，仍失败的文件记为"待重试"，下次扫描再处理；其他错误跳过该文件，不影响其余文件。

### 3.6 检索流程
1. 处理检索前先做**新鲜度检查**（见第 4 节），把脏文件同步更新完，或在超时预算内尽量更新。
2. `encode_queries([query])` → 与 VectorCache 做矩阵乘法 → `argpartition` 取 `k * 6` 个候选。
3. 过滤：`path_glob`（gitignore 语法，支持 `{a,b}` 和 `!` 排除）、`modality`、`min_score`，在排序前生效。`path_glob` 没有匹配任何已索引文件时，结果里附一条说明。
4. 去重：内容完全相同的块合并成一条，并列出其他副本路径；每个文件最多保留 2 个块，与已选块行区间重叠的直接丢弃（不合并）。重复太多、结果不够 k 条时逐步扩大候选池，最多 5000。
5. 默认纯向量检索。另有可选的混合检索（阶段 1.5）：查询里有标识符形状的词时，FTS5 BM25 结果按"余弦 + 0.05 × 归一化 BM25"并入排序。分块器 v4 之后，在 126 道评测题上混合检索与纯向量持平（MRR 0.774 对 0.778），所以默认不启用，见 `docs/EVAL_REAL_PROJECTS.md`。`SEMSEARCH_SEARCH_MODE=vector|keyword|hybrid` 可切换；关键词索引照常建立。
6. 返回 top-k：

```json
{
  "results": [
    {
      "path": "src/net/retry.py",
      "lines": [42, 88],
      "symbol": "RetryPolicy.backoff",
      "kind": "method",
      "modality": "code",
      "score": 0.71,
      "preview": "def backoff(self, attempt): ...",
      "duplicates": []
    }
  ],
  "fresh": true,
  "status": {"files": 1834, "chunks": 12410, "indexing": false, "...": "..."}
}
```

- PDF 结果多一个 `page`，音视频结果多 `t_start` / `t_end`（秒）。
- MCP 和 CLI 把它格式化成文本：路径、符号名去掉换行、控制字符和不可见字符并限制长度，预览加 `|` 前缀，整体标注为不可信的文件内容。

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
files_now  = scanner.list()                      # git ls-files / rg --files / 遍历
stat 比较 files 表：
  新文件 / size 或 mtime 变化 → 读内容算 content_hash
      hash 未变 → 只更新 mtime（例如 touch）
      hash 变化 → 重新分块：
          新块 hash 已存在于库 → 复用向量
          否则 → 加入嵌入批次
          旧块中不再出现的 → 删除（chunks + vec + fts）
  files 表有但磁盘已无 → 删除文件及其块
批量嵌入 → 每 256 个块提交一次事务 → 提交成功后更新 VectorCache
```
- 编辑一个函数，通常只重新嵌入 1–2 个块。
- 首次索引在后台进行，期间检索返回已完成部分，并以 `fresh=false` 和 `status.progress` 标明进度。
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
| `semantic_search` | `query`, `k=8`, `path_glob?`（逗号分隔，花括号内的逗号不拆）, `modality?` | 主要入口；`min_score` 只在 HTTP 接口和 CLI 上提供 |
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
        "hooks": [{ "type": "command", "command": "\"<uv 工具目录>/semsearch-hook\"", "timeout": 5 }]
      }
    ]
  }
}
```
- 从 stdin 按 UTF-8 读取 hook JSON（不受系统代码页影响），取出 `tool_input.file_path`，POST 到 `/notify`。项目根优先取 `CLAUDE_PROJECT_DIR`。
- 通知请求超时 0.5 秒；失败不影响 Claude 的操作，但写入 `~/.cache/semsearch/logs/hook.log`（按大小轮转）（不静默吞错误）。
- daemon 未运行时什么也不做（只记日志），不为一次 hook 拉起模型；下次检索前的快速扫描会补上这次改动。
- daemon 只更新**已经建过索引**的项目：在从没用过语义检索的项目里编辑文件，不会触发整个项目的首次索引。

### 5.3 Subagent：`~/.claude/agents/semantic-search.md`
- 实际内容见 `src/semsearch/integrations/semantic-search.md`，由安装器复制过去（已有文件先备份）。
- 工具：`mcp__semsearch__semantic_search, Read, Grep, Glob`，模型 Haiku。
- 流程：先把问题翻译成英文检索，必要时换 1–2 种说法再查；用 Read 核实靠前的结果；找到确切符号后用 Grep 补全定义与主要引用；工具调用次数有上限。
- 输出：只返回核实过的 `path:start-end` — 符号名 — 一句话说明；没有可信结果时明确说"未找到"，并给出建议的 Grep 关键词。

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
semsearch install-claude             # 注册 MCP、写入 hook、subagent、CLAUDE.md 路由片段（--dry-run 只展示改动）
```
- 安装器改 JSON 配置时，写入前重新读取并合并一次，避免覆盖 Claude Code 同时写入的内容；每个文件留 `.semsearch.bak`（第一次改动前）和 `.semsearch.prev`（最近一次改动前）两份备份。
- 更新代码后：`uv tool install . --python 3.12 --reinstall`，再 `semsearch daemon stop`，让新代码生效。仓库的 post-commit hook 会在提交后自动做这两步（见 README 开发一节）。
- 卸载：`semsearch install-claude --uninstall`，然后 `uv tool uninstall semsearch`。
- 写入的配置指向 `%APPDATA%\uv\tools\semsearch\Scripts\` 下的绝对路径，不依赖 PATH。

CLI 也可单独使用：`semsearch search "限流重试"`、`semsearch status`、`semsearch reindex`。

---

## 6. 项目结构

```
SearchEngine/
├─ pyproject.toml / uv.lock
├─ .githooks/post-commit     # 提交后从 HEAD 快照重装 uv tool，并停掉旧 daemon
├─ docs/                     # DESIGN、PHASE*、EVAL_REAL_PROJECTS、AUDIT
├─ src/semsearch/
│  ├─ config.py           # Settings（环境变量）、扩展名、敏感文件规则、版本号
│  ├─ paths.py            # 项目根查找、project_key、拒绝的根目录
│  ├─ permissions.py      # 数据目录只允许当前用户访问（Windows ACL / POSIX 700）
│  ├─ globs.py            # 不可信 glob 的规范化与 RE2 匹配
│  ├─ formatting.py       # 结果转文本、清理不可信内容
│  ├─ daemon/
│  │  ├─ __main__.py      # 启动、daemon.json、空闲退出
│  │  ├─ app.py           # FastAPI 路由、鉴权、项目注册
│  │  └─ limits.py        # 请求体上限、令牌桶限速
│  ├─ embed/
│  │  ├─ base.py          # Embedder 协议、FakeEmbedder
│  │  └─ st_backend.py    # sentence-transformers 实现、加载模式
│  ├─ index/
│  │  ├─ scanner.py       # 列文件、忽略规则、敏感文件
│  │  ├─ chunkers/        # code.py / markdown.py / window.py / base.py
│  │  ├─ media.py         # 图片、PDF、音视频的分段与解码
│  │  ├─ media_worker.py  # 媒体解码子进程（限制内存/时间）与回传数据检查
│  │  ├─ indexer.py       # 增量更新：分块、复用向量、嵌入、写入
│  │  ├─ project.py       # 单个项目：写者线程、扫描、检索入口、状态
│  │  ├─ store.py         # SQLite schema 与读写
│  │  ├─ vectors.py       # VectorCache
│  │  ├─ keywords.py      # 关键词分词（标识符拆分）
│  │  └─ search.py        # KNN、关键词加分、过滤、去重
│  ├─ integrations/       # semantic-search.md（subagent）、claude-md-snippet.md
│  ├─ client.py           # 连接/拉起 daemon 的共享客户端
│  ├─ install.py          # install-claude
│  ├─ mcp_server.py
│  ├─ hook.py
│  └─ cli.py
├─ evals/
│  ├─ cases/<repo>.json      # {project, aliases, cases: [{q, tags, expect: [{path, symbol|lines}]}]}
│  ├─ private/               # 私人项目的题目（gitignore，不公开）
│  ├─ index_projects.py      # 预先建好评测用索引
│  └─ run.py
└─ tests/                    # 单元 / 集成 / 端到端；conftest.py 隔离环境变量
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

**结果**（阶段 1 结束时）：102 个测试通过（另有 2 个按条件跳过：真模型 GPU 测试、需要开发者模式的符号链接测试），覆盖率 91%，ruff / pyright 无问题。真实项目（RaftMod-Publish，235 个文件，3,076 块）首次索引约 82 秒，热查询约 0.7 秒（CLI，含进程启动），冷启动约 19 秒。改一个文件只重新嵌入变化的块（实测 1 块）。

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

### 阶段 1.5：检索质量（已完成，见 `docs/PHASE1_5.md`）
- [x] 评测集（2 个项目共 54 条）与 `evals/run.py`
- [x] FTS5 关键词索引 + 混合检索（只对含标识符的查询加 BM25 分；RRF 评测更差，未采用）
- [x] 中文查询：先翻译成英文再查（盲译测试中，中文类 MRR 0.58 → 0.70），写进工具说明、subagent 与路由规则；查询含中日韩文字时，结果末尾提示改用英文。换提示词、放宽每文件条数都没有明显效果
- [x] 真实项目评测（6 个私人项目 72 题，见 `docs/EVAL_REAL_PROJECTS.md`）：分块器 v4 让紧贴在定义前的注释并入该定义的块，并识别 PowerShell 函数；MRR 从 0.65 升到 0.81
- [x] 嵌入模型可配置：用 `SEMSEARCH_MODEL / EMBED_DIM / QUERY_PROMPT / DOCUMENT_PROMPT / CHUNK_TOKENS / MAX_TOKENS / TRUST_REMOTE_CODE` 切换任意 sentence-transformers 模型。提示词默认自动选择：EmbeddingGemma 用 SearchQuery/Document，其他模型用其配置里的 query/document，都没有就不加。`max_tokens` 不超过模型自身的上限。维度超出时，在比对索引元数据之前就报错，避免误清空索引。索引元数据新增 `document_prompt` 和 `chunk_tokens` 两项，变化时自动重建；只改查询提示词不重建
- [ ] 调块大小、块头格式、每文件结果数（用评测集验证）

**结果**：MRR 0.698 → 0.690（基本持平），R@5 0.796 → 0.815。关键词索引约占 8% 空间，查询约 1 ms。

### 阶段 2：图片与 PDF（已完成，见 `docs/PHASE2_3.md`）
- [x] 视觉编码器按需加载：由 `SEMSEARCH_MEDIA` 决定加载哪些编码器（load_mode 为 text / text+vision / text+audio / full），load_mode 写入索引元数据
- [x] 图片：每张一块，只用像素；小于 32×32 的图标跳过
- [x] PDF：每页一块，页面文字和渲染图像合成一个向量；结果显示 `file.pdf p.N`，页面文字同时进入关键词索引
- [x] 结果返回页码和图片路径

### 阶段 3：音频与视频（已完成）
- [x] 音频：重采样为 16kHz 单声道，按 30 秒窗口、5 秒重叠切段
- [x] 视频：每秒抽 1 帧，按 20 秒切段；启用音频编码器时，该段的声音也一起编码
- [x] 结果返回时间段 `file mm:ss-mm:ss`

**结果**：30 题的媒体评测中，图片 MRR 0.94，答案进前 5 的比例 100%；PDF 和视频进前 5 的比例为 83%；整体 MRR 0.77，进前 5 的比例 93%。

### 全局审校（已完成，见 `docs/AUDIT.md`）
- [x] 编码、daemon 启动、索引一致性（批次 1）
- [x] 安全：ReDoS、数据目录权限、拒绝的根目录、敏感文件大小写（批次 2）
- [x] 检索与分块：分块器 v5、候选池扩大、媒体去重（批次 3）
- [x] 测试、打包、文档（批次 4）

---

## 8. 评估方法
- 选 2–3 个真实仓库（包含你自己的项目和一个开源中型仓库，如 HF transformers 的子目录）。
- 每个仓库写 20–30 条查询，分三类：
  - 概念类："哪里做请求重试"
  - 描述类："把配置从环境变量读进来的地方"
  - 相似类：给一段代码，找类似实现
- 指标：Recall@5、Recall@10、MRR；同时记录"用 grep 需要几次尝试才能找到"作为对照。
- 每次调整分块或检索策略都跑一遍，结果写入 `evals/results/`。
- 已实现：`uv run python evals/run.py --modes vector,hybrid --show-misses`（使用独立索引目录 `evals/.home`）。

---

## 9. 风险与待确认事项
| 项 | 风险 | 处理 |
|---|---|---|
| 中文查询召回偏弱 | 关键词检索帮不上（中文描述对英文代码） | 已处理：让调用方先翻译成英文（分块器 v4 之后中文题 MRR 0.77 → 0.80），结果中附提示；是否遵守取决于 Claude |
| 媒体解码 | PyAV / pdfium / Pillow 解析不可信文件 | 默认关闭媒体索引；有大小、页数、时长、像素上限；在受限子进程里解码，回传数据逐项检查（见 3.4 和 `docs/AUDIT.md` 批次 5） |
| 大型 C# 项目首次索引慢 | 1 万多个文件可能要 20–30 分钟 | 后台索引、返回部分结果并标注进度；阶段 1 尝试按 token 预算组 batch 提速 |
| 6GB 显存 | 全模态 + 大批量可能 OOM | 自适应批大小；默认只加载文本编码器；空闲释放 |
| Windows 兼容 | 路径分隔符与大小写（依赖兼容性已在阶段 0 验证） | 路径统一规范化为小写盘符 + `/` |
| 大仓库首次索引耗时 | 几十万行代码可能需要数分钟 | 后台索引、检索返回部分结果并标注进度 |
| 敏感信息 | 索引片段会进入 Claude 上下文 | 默认忽略密钥类文件；支持 `.semsearchignore` |
| 本机其他进程访问 daemon | 可读取项目内容 | 仅绑定 127.0.0.1 + token；拒绝项目根之外的路径 |
| 多会话并发 | 同一项目多处同时触发更新 | 每项目单写者队列，SQLite WAL |

---

## 10. 依赖清单
- 运行时：`torch` + `torchvision`（Windows/Linux 用 cu130）、`pillow`、`sentence-transformers>=6.1`、`transformers`、`numpy`、`tree-sitter`、`tree-sitter-language-pack`、`blake3`、`pathspec>=1.1`、`google-re2`、`fastapi`、`pydantic>=2`、`uvicorn`、`httpx`、`anyio`、`mcp>=2,<3`
- 媒体：`pypdfium2`（PDF 文字与渲染）、`av`（PyAV，音视频解码与重采样）
- 开发：`pytest`、`pytest-cov`、`ruff`、`pyright`（`sqlite-vec` 只用于阶段 0 的对比脚本）
- 外部工具：`git`、`ripgrep`（都可选；都没有时自己遍历目录）
- 版本以 `pyproject.toml` 和 `uv.lock` 为准。
