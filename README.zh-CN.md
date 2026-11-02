# semsearch

[English](README.md)

基于 [EmbeddingGemma 2](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/) 的本地语义代码检索，给 [Claude Code](https://claude.com/claude-code) 用。

Claude Code 平时用 Grep 和 LSP 找代码，前提是知道名字。semsearch 用在只知道**代码做什么**、不知道名字的时候，比如"下载失败后在哪重试"、"防止崩溃把存档写坏的逻辑"、"和这段相似的代码"。它是 Grep / LSP 的补充，不替代它们。

- **完全本地**：模型由一个后台服务统一加载，索引存在 `~/.cache/semsearch`。
- **增量更新**：只重新处理改动过的文件，只重新嵌入变化的代码块。Claude 每次编辑后由 hook 通知更新；在编辑器里保存、`git pull`、切分支，会在下次检索前扫描出来。
- **按语法分块**：用 tree-sitter 按函数、方法、类切分代码，文档注释和它描述的代码放在同一块；Markdown 按标题切分。
- **接入 Claude Code**：提供 MCP 工具、`semantic-search` 子代理、编辑 hook 和工具路由规则，Claude 会自己判断什么时候用。

## 效果
数字来自仓库里的评测集（`evals/`）。题目由没看过实现的子代理编写，每个答案都核对过源码。

| 评测集 | 题数 | MRR | 答案进前 5 |
|---|---|---|---|
| 6 个真实游戏 mod 项目（C#、TS、GDScript、PowerShell、JS；最大 6 万个块） | 72 | 0.83 | 96% |
| 1 个公开的 C# mod + 本仓库 | 54 | 0.76 | 87% |

测试方式与 Claude 实际使用时一致：非英文问题先翻译成英文再查。评测方法和试过的全部方案见 [docs/EVAL_REAL_PROJECTS.md](docs/EVAL_REAL_PROJECTS.md) 与 [docs/PHASE1_5.md](docs/PHASE1_5.md)。

速度（RTX 3060 6GB）：
- 已加载模型时查询约 0.7 秒（含 CLI 启动）；后台服务启动后的第一次查询约 20 秒，要先加载模型。
- 首次索引：250 个文件约 1.5 分钟；8,000 个文件（6 万块）约 26 分钟。

## 环境要求
- **Python 3.12** 和 [uv](https://docs.astral.sh/uv/)。
- **推荐 NVIDIA 显卡**：没有显卡会退回 CPU，能用但慢很多。
- **PyTorch**：`pyproject.toml` 安装的是 CUDA 13.0 版（`pytorch-cu130`）。其他 CUDA 版本或纯 CPU，改 `[tool.uv.sources]` / `[[tool.uv.index]]`。
- **系统**：在 Windows 11 上开发和测试。代码本身跨平台，Linux 和 macOS 尚未测试。
- **模型**：`google/embeddinggemma-2` 在第一次使用时从 Hugging Face 下载。

## 安装
```bash
git clone https://github.com/spardanviro/SearchEngine.git
cd SearchEngine
uv tool install . --python 3.12
semsearch install-claude --dry-run   # 先看会改哪些地方
semsearch install-claude             # 注册到 Claude Code（执行前会确认）
```
`uv tool install` 会把 semsearch 复制到独立环境，之后移动或删除源码目录都不影响使用。

`install-claude` 只改用户级配置，改之前先备份原文件（`*.semsearch.bak`）：

| 文件 | 改动 |
|---|---|
| `~/.claude.json` | 注册 `semsearch` MCP 服务 |
| `~/.claude/settings.json` | 添加 PostToolUse hook，编辑文件后通知后台服务 |
| `~/.claude/agents/semantic-search.md` | 子代理定义 |
| `~/.claude/CLAUDE.md` | 一段带标记的工具路由规则 |

装好后开一个新的 Claude Code 会话即可生效。卸载：
```bash
semsearch install-claude --uninstall
uv tool uninstall semsearch
```

## 使用
### 在 Claude Code 里
一般什么都不用做，照常提问，Claude 会自己选工具：

| 你知道的 | Claude 用 |
|---|---|
| 确切的名字、字符串、正则 | Grep / Glob |
| 某个符号，要找定义、引用、调用者 | LSP |
| 只知道代码做什么 | `semantic_search` |
| 开放式问题，需要多次查找并读代码核实 | `semantic-search` 子代理 |

也可以直接点名，比如"用语义检索找一下……"或"让 semantic-search 子代理查查……"。

在某个项目里第一次检索时，会在后台为它建索引。建好之前，结果会标注"可能不完整"，Claude 会改用 Grep。

### 命令行
```bash
semsearch search "第二位玩家倒地后怎么复活" -k 5
semsearch search "save file backup" -g "src/**" -m code --json
semsearch status
semsearch reindex            # 立即重新扫描
semsearch reindex --full     # 从头重建索引
semsearch daemon stop        # 释放显存；下次检索会自动重启
```
加 `--root <目录>` 可以查别的项目。命令行下用英文写查询效果更好；中文也能用，只是排序会差一些。

### 排除文件
- `.gitignore` 排除的文件不会被索引。
- 还想排除别的，在项目根目录建 `.semsearchignore`，写法和 `.gitignore` 相同。
- `.env*`、`*.pem`、`*.key`、`id_rsa*`、凭据文件等敏感文件一律跳过，忽略规则里写 `!` 也不能把它们加回来。

## 配置（环境变量）
| 变量 | 默认值 | 说明 |
|---|---|---|
| `SEMSEARCH_HOME` | `~/.cache/semsearch` | 索引、日志、后台服务状态 |
| `SEMSEARCH_IDLE_TIMEOUT` | `1200` | 后台服务空闲多少秒后退出 |
| `SEMSEARCH_SEARCH_MODE` | `vector` | `vector`、`hybrid`（对含标识符的查询加 BM25 分）、`keyword` |
| `SEMSEARCH_KEYWORD_WEIGHT` | `0.05` | `hybrid` 模式下 BM25 加分的上限 |
| `SEMSEARCH_MAX_FILE_BYTES` | `1000000` | 超过这个大小的文件跳过 |
| `SEMSEARCH_MAX_FILES` | `50000` | 每个项目最多索引的文件数 |
| `SEMSEARCH_BATCH_TOKENS` | `16384` | 每批嵌入的 token 上限，显存不够时调小 |

## 原理
```
Claude Code ── MCP (stdio) ──┐
semsearch 命令行 ────────────┼── HTTP（127.0.0.1 + token）──> semsearchd
PostToolUse hook ────────────┘                               ├─ EmbeddingGemma 2（只加载一份）
                                                             ├─ 每个项目一个 SQLite 索引（块、向量、FTS5）
                                                             └─ 内存里的向量矩阵
```
- 一个后台服务处理所有项目：用到时启动，空闲后退出。
- **存储与检索**：数据以 SQLite（WAL 模式）为准，向量以 BLOB 存储。检索在 numpy 里做暴力矩阵乘法，这个规模下比 sqlite-vec 快约 30 倍。
- **变更检测**：先比 mtime 和文件大小，再比 blake3 内容哈希。按块哈希复用向量，搬动或复制的代码不会重新计算嵌入。
- **重复代码**：复制出来的相同代码合并成一条结果，并注明其他副本的位置。
- **安全**：返回给 Claude 的预览会去掉换行、控制字符等可能伪造结果的内容，并标注为不可信的文件内容。

完整设计见 [docs/DESIGN.md](docs/DESIGN.md)。

## 开发
```bash
uv sync
git config core.hooksPath .githooks   # 每次提交后自动更新已安装的版本
uv run pytest -q
uv run ruff check src tests evals
pyright src tests evals
```
提交改动了 `src/`、`pyproject.toml` 或 `uv.lock` 时，post-commit hook 会重新安装 uv tool 里的副本，保证 Claude Code 用的总是已提交的代码。

评测：
```bash
uv run python evals/index_projects.py <项目根目录> ...
uv run python evals/run.py --modes "vector,hybrid,hybrid;tr=1" --show-misses
```
- 题目放在 `evals/cases/`，格式参考 `semsearch.json`。
- 私人项目的题目放 `evals/private/cases/`：这个目录不会被提交，评测时会自动读取。

## 计划
- 图片和 PDF（EmbeddingGemma 2 支持多模态）
- 音频和视频

## 许可证
本项目代码采用 [MIT](LICENSE) 许可证。EmbeddingGemma 2 模型不包含在仓库中，需要单独下载；模型的使用条款以 Google 在 Hugging Face 上发布的为准，不受 MIT 许可证约束。
