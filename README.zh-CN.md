# semsearch

[English](README.md)

基于 [EmbeddingGemma 2](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/) 的本地语义代码检索，给 [Claude Code](https://claude.com/claude-code) 用。

Claude Code 平时用 Grep 和 LSP 找代码，前提是知道名字。semsearch 用在只知道**代码做什么**、不知道名字的时候，比如"下载失败后在哪重试"、"防止崩溃把存档写坏的逻辑"、"和这段相似的代码"。它是 Grep / LSP 的补充，不替代它们。

- **完全本地**：模型由一个后台服务统一加载，索引存在 `~/.cache/semsearch`。
- **增量更新**：只重新处理改动过的文件，只重新嵌入变化的代码块。Claude 每次编辑后由 hook 通知更新；在编辑器里保存、`git pull`、切分支，会在下次检索前扫描出来。
- **按语法分块**：用 tree-sitter 按函数、方法、类切分代码，文档注释和它描述的代码放在同一块；Markdown 按标题切分。
- **接入 Claude Code**：提供 MCP 工具、`semantic-search` 子代理、编辑 hook 和工具路由规则，Claude 会自己判断什么时候用。
- **图片、PDF、音频、视频（可选）**：同样可以按内容检索，比如"蓝色爆炸序列帧"、"有体素可见性示意图的那一页"、"角色倒地瘫软的那段视频"。详见下方[媒体检索](#媒体检索)。

## 效果
| 评测集 | 题数 | MRR | 答案进前 5 |
|---|---|---|---|
| 6 个私人游戏 mod 项目（C#、TS、GDScript、PowerShell、JS；最大 6 万个块） | 72 | 0.83 | 96% |
| 1 个公开的 C# mod + 本仓库（`evals/cases/`） | 54 | 0.79 | 91% |

- **题目来源**：72 题由没看过实现的子代理编写，每个答案都核对过源码；由于对应项目是私人项目，这些题只保存在作者本地，没有公开。54 题在本仓库里，是开发者自己写的，可能比实际使用时简单一些；其中关于本仓库的题目会随代码改动而漂移，代码搬动位置后要同步更新标准答案（见 [docs/AUDIT.md](docs/AUDIT.md)）。
- **测试方式**：与 Claude 实际使用时一致，非英文问题先翻译成英文再查。表中分数是 hybrid 模式的结果；默认的纯向量模式在 72 题上相差不到 0.01 MRR，在 54 题上低 0.03。
- **详细说明**：评测方法和试过的全部方案见 [docs/EVAL_REAL_PROJECTS.md](docs/EVAL_REAL_PROJECTS.md) 与 [docs/PHASE1_5.md](docs/PHASE1_5.md)。

速度（RTX 3060 6GB）：
- 已加载模型时查询约 0.7 秒（含 CLI 启动）；后台服务启动后的第一次查询约 20 秒，要先加载模型。
- 首次索引：250 个文件约 1.5 分钟；8,000 个文件（6 万块）约 26 分钟。

## 环境要求
- **Python 3.12** 和 [uv](https://docs.astral.sh/uv/)。
- **推荐 NVIDIA 显卡**：没有显卡会退回 CPU，能用但慢很多。
- **PyTorch**：Windows 和 Linux 上，`pyproject.toml` 安装 CUDA 13.0 版（`pytorch-cu130`），需要 580 及以上版本的 NVIDIA 驱动，驱动太旧时会静默退回 CPU（`semsearch doctor` 能检查出来）。要用其他 CUDA 版本或纯 CPU，改 `[tool.uv.sources]` / `[[tool.uv.index]]`。macOS 使用 PyPI 上的普通版本。
- **系统**：在 Windows 11 上开发和测试。代码本身跨平台，Linux 和 macOS 尚未测试。
- **模型**：`google/embeddinggemma-2`（1.5 GB）在第一次使用时从 Hugging Face 下载，也可以先用 `semsearch download-model` 提前下载（会先询问）。

## 安装
### 一键安装（推荐）
Windows（PowerShell）：
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/spardanviro/semsearch/releases/latest/download/install.ps1 | iex"
```
macOS / Linux：
```bash
curl -LsSf https://github.com/spardanviro/semsearch/releases/latest/download/install.sh | sh
```
脚本会依次：没有 [uv](https://docs.astral.sh/uv/) 时安装 uv；按 `uv.lock` 锁定的依赖版本安装 semsearch；运行 `semsearch doctor` 检查环境；询问是否下载模型；最后运行 `semsearch install-claude` 注册到 Claude Code。每次下载、每处改动都会先询问。需要约 5 GB 磁盘空间（PyTorch 约 3.5 GB，模型 1.5 GB）。Windows 脚本在 Windows 11 上测试过；`install.sh` 目前只在 Windows 的 Git Bash 里跑过。

可用环境变量调整：`SEMSEARCH_INSTALL_REF`（分支、标签或提交，默认是脚本所属的 release 版本）、`SEMSEARCH_INSTALL_SOURCE`（从本地目录安装）、`SEMSEARCH_INSTALL_YES=1`（所有询问都回答"是"）。

### 手动安装
```bash
git clone https://github.com/spardanviro/semsearch.git
cd semsearch
uv tool install . --python 3.12
semsearch install-claude --dry-run   # 先看会改哪些地方
semsearch install-claude             # 注册到 Claude Code（执行前会确认）
```
`uv tool install` 会把 semsearch 复制到独立环境，之后移动或删除源码目录都不影响使用。

`install-claude` 只改用户级配置：

| 文件 | 改动 |
|---|---|
| `~/.claude.json` | 注册 `semsearch` MCP 服务 |
| `~/.claude/settings.json` | 添加 PostToolUse hook，编辑文件后通知后台服务 |
| `~/.claude/agents/semantic-search.md` | 子代理定义 |
| `~/.claude/CLAUDE.md` | 一段带标记的工具路由规则（会替换已有的 `# 代码导航` 一节） |

每个被改动的文件都留两份备份：
- `*.semsearch.bak`：semsearch 第一次改动它之前的样子；
- `*.semsearch.prev`：最近一次改动之前的样子。

备份是原文件的完整副本，例如 `~/.claude.json` 里的 token 也会原样出现在 `~/.claude.json.semsearch.bak` 里。

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
semsearch doctor             # 检查 Python、显卡和驱动、模型、磁盘、Claude Code 注册情况
semsearch download-model     # 现在就下载模型（先显示大小并询问）
```
`semsearch doctor` 会为每个问题给出修复方法，例如"NVIDIA 驱动太旧，正在用 CPU 运行"、"连不上 Hugging Face"。只有模型还没下载时才会联网查询，加 `--offline` 可跳过。
加 `--root <目录>` 可以查别的项目。命令行下用英文写查询效果更好；中文也能用，只是排序会差一些。

### 排除文件
- `.gitignore` 排除的文件不会被索引。
- 还想排除别的，在项目根目录建 `.semsearchignore`，写法和 `.gitignore` 相同。
- `.env*`、`*.pem`、`*.key`、`id_rsa*`、凭据文件等敏感文件一律跳过：不区分大小写，任何目录层级都生效，忽略规则里写 `!` 也不能把它们加回来。完整列表见 `src/semsearch/config.py` 里的 `SENSITIVE_PATTERNS`。
- 压缩过的打包文件也跳过：平均每行内容超过 500 个字符的文件不会被索引。
- 以下目录不会被当作项目索引：用户主目录本身、磁盘根目录、`~/.claude`、`~/.ssh`。请在具体的项目文件夹里打开 Claude Code。

## 媒体检索
EmbeddingGemma 2 把文字、图片、音频、视频映射到同一个向量空间，所以用一句文字就能搜到这些媒体。媒体检索**默认关闭**，用下面的命令开启：
```bash
setx SEMSEARCH_MEDIA "image,pdf"     # 也可以写 image,pdf,audio,video，或直接写 all
semsearch daemon stop
```

| 类型 | 一块是什么 | 结果中显示为 |
|---|---|---|
| image（png、jpg、webp、bmp、gif） | 整张图片（小于 32×32 的图标跳过） | `art/hero.png` |
| pdf | 一页：页面文字加页面渲染图 | `docs/manual.pdf p.12` |
| audio（wav、mp3、ogg、flac、m4a 等） | 30 秒一段，相邻两段重叠 5 秒 | `sfx/theme.ogg 01:05-01:35` |
| video（mp4、webm、mkv、mov 等） | 20 秒一段，每秒取 1 帧；开启音频时连同这段的声音一起编码 | `clips/intro.mp4 00:40-01:00` |

- 只加载已开启的类型需要的编码器，全部开启时约占 1.4 GB 显存。开启媒体检索或改动类型列表后，每个项目的索引会自动重建一次。
- 图片和 PDF 的结果，Claude 可以直接用 Read 打开；PDF 结果会注明页码。音频和视频的结果只给出时间段。
- 评测（30 题，素材来自 Godot 文档里的插图和视频，外加一篇论文）：图片 MRR 0.94，正确答案每次都在前 5；PDF 页和视频片段 83% 的题答案在前 5。详见 [docs/PHASE2_3.md](docs/PHASE2_3.md)。
- 游戏项目里可能有成千上万张贴图，用不到的素材目录建议在 `.semsearchignore` 里排除。
- 限制：单个文件最大 `SEMSEARCH_MAX_MEDIA_BYTES`（默认 500 MB）；每个 PDF 最多 `SEMSEARCH_MAX_PDF_PAGES` 页（默认 500 页）；每个音视频最多索引前 `SEMSEARCH_MAX_MEDIA_SECONDS` 秒（默认 3600 秒），超出部分不索引。

## 配置（环境变量）
| 变量 | 默认值 | 说明 |
|---|---|---|
| `SEMSEARCH_HOME` | `~/.cache/semsearch` | 索引、日志、后台服务状态 |
| `SEMSEARCH_IDLE_TIMEOUT` | `1200` | 后台服务空闲多少秒后退出 |
| `SEMSEARCH_SEARCH_MODE` | `vector` | `vector`、`hybrid`（对含标识符的查询加 BM25 分）、`keyword` |
| `SEMSEARCH_KEYWORD_WEIGHT` | `0.05` | `hybrid` 模式下 BM25 加分的上限 |
| `SEMSEARCH_MAX_FILE_BYTES` | `1000000` | 超过这个大小的源码和文档文件跳过（json、yaml、xml 等配置文件的上限固定为 200 KB） |
| `SEMSEARCH_MAX_FILES` | `50000` | 每个项目最多索引的文件数 |
| `SEMSEARCH_BATCH_TOKENS` | `16384` | 每批嵌入的 token 上限，显存不够时调小 |
| `SEMSEARCH_MODEL` | `google/embeddinggemma-2` | 任意 sentence-transformers 模型名或本地路径 |
| `SEMSEARCH_EMBED_DIM` | `0` | 向量维度；`0` 表示用模型自己的维度，设得更小会截断（适用于 Matryoshka 类模型） |
| `SEMSEARCH_QUERY_PROMPT` | 自动 | 查询提示词：模型卡里的提示词名，或直接写前缀文本（`""` 表示不加） |
| `SEMSEARCH_DOCUMENT_PROMPT` | 自动 | 文档提示词，规则同上，作用于被索引的代码块 |
| `SEMSEARCH_CHUNK_TOKENS` | `600` | 代码块的目标大小（token） |
| `SEMSEARCH_MAX_TOKENS` | `1024` | 单块的硬上限；超过模型自身上限时自动压到模型上限，并给出警告 |
| `SEMSEARCH_MODEL_REVISION` | 固定版本 | Hugging Face 上的模型版本；默认模型固定在测试过的那个 commit，其他模型用仓库默认分支 |
| `SEMSEARCH_TRUST_REMOTE_CODE` | `false` | 是否允许运行模型仓库自带的 Python 代码 |
| `SEMSEARCH_MEDIA` | （关闭） | 要索引的媒体类型：`image`、`pdf`、`audio`、`video`，或 `all` |
| `SEMSEARCH_MAX_MEDIA_BYTES` | `500000000` | 超过这个大小的媒体文件跳过 |
| `SEMSEARCH_MAX_PDF_PAGES` | `500` | 每个 PDF 最多索引的页数 |
| `SEMSEARCH_MAX_MEDIA_SECONDS` | `3600` | 每个音视频文件最多索引的时长（秒） |
| `SEMSEARCH_EMBEDDER` | `st` | 嵌入后端：`st`（sentence-transformers）；`fake` 只供测试使用 |

两个提示词变量都不设时，程序按以下顺序自动选择：
- EmbeddingGemma：用它自带的 `SearchQuery` / `Document`；
- 其他模型：如果它的 sentence-transformers 配置里定义了 `query` / `document`，就用这两个；
- 以上都没有：不加提示词。

后台服务只在启动时读取这些配置。改完要执行 `semsearch daemon stop`，下次检索时服务会按新配置重新启动。这些变量要设成**用户环境变量**，这样不管后台服务是由 Claude Code 还是命令行拉起的，都能读到。

### 换模型
sentence-transformers 能加载的嵌入模型，都可以直接换上，不用改代码。例如：
```bash
# Windows：setx SEMSEARCH_MODEL "Qwen/Qwen3-Embedding-0.6B"（设完要重开终端）
export SEMSEARCH_MODEL="Qwen/Qwen3-Embedding-0.6B"   # 自带 query/document 提示词，自动生效
semsearch daemon stop
```
- **提示词**：每个模型的查询和文档提示词都不一样，写错会明显拉低排序质量。如果模型的 sentence-transformers 配置里没有定义 `query` / `document`，就照模型卡手动设置 `SEMSEARCH_QUERY_PROMPT` / `SEMSEARCH_DOCUMENT_PROMPT`，例如 E5 系列用 `"query: "` 和 `"passage: "`。填了模型不认识的提示词名时，会按字面文本使用，后台服务日志里会有警告。
- **上下文长度**：如果模型支持的上下文短于 1024 token，把 `SEMSEARCH_MAX_TOKENS` 和 `SEMSEARCH_CHUNK_TOKENS` 一起调小。不调的话会按模型自身上限截断，长代码块的末尾会丢失。
- **自动重建**：模型名、向量维度、文档提示词和块大小都记录在索引里，任何一项改变都会自动重建索引。只改查询提示词不会触发重建。
- **`SEMSEARCH_TRUST_REMOTE_CODE`**：有些模型需要打开它。打开后会运行从模型仓库下载的 Python 代码，只对信得过的发布者开启。
- **先评测再换**：用评测集在你自己的代码上比较新旧模型。给每个模型单独指定索引目录，避免互相覆盖：`uv run python evals/run.py --home evals/.home-<模型名> ...`。
- **其他后端**：想接 Ollama 或在线 API，在 `src/semsearch/embed/base.py` 里实现 `Embedder` 接口，再在 `make_embedder` 里注册（对应 `SEMSEARCH_EMBEDDER`）。用在线 API 时，代码会被发送到你的电脑之外。

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
- **不可信内容**：路径、符号名和预览都来自仓库文件。返回给 Claude 之前，会去掉换行、控制字符和不可见字符，限制长度，并标注为不可信内容。
- **私有数据**：后台服务只监听 127.0.0.1，访问需要 token。它的数据目录（token、索引和日志）只允许当前用户读取；在 Windows 上通过显式的 ACL 实现。请求体上限 1 MB，并有频率限制。
- **媒体文件**：图片、PDF、音视频在单独的子进程里解码，有内存上限和时间限制（Windows 上还禁止它启动其他进程）。某个文件让它崩溃或卡死时，只影响这个文件。子进程以你的用户身份运行，所以它限制的是损害范围，不是完整的沙箱。

完整设计见 [docs/DESIGN.md](docs/DESIGN.md)。

## 开发
```bash
uv sync
git config core.hooksPath .githooks   # 每次提交后自动更新已安装的版本
uv run pytest -q
uv run ruff check src tests evals
uv run pyright src tests evals
```
- **真实模型测试**：默认跳过，设 `SEMSEARCH_GPU_TESTS=1` 后运行 `uv run pytest -q -m gpu`。
- **测试隔离**：测试会忽略你自己设置的 `SEMSEARCH_*` 环境变量。
- **CI**：GitHub Actions 在 Windows、Linux、macOS 上运行上面这些检查；还会在每个系统的干净机器上运行一键安装脚本，再用真实模型在 CPU 上为一个小项目建索引并检验搜索结果（`.github/scripts/smoke.py`）。
- **提交后自动同步**：提交改动了 `src/`、`pyproject.toml` 或 `uv.lock` 时，post-commit hook 会从 `HEAD` 的快照重新安装 uv tool 里的副本，依赖版本按 `uv.lock` 锁定，未提交的改动不会进入 Claude Code。
- **不会自动同步的情况**：pull、checkout、rebase 不会触发这个 hook，之后要手动执行 `uv tool install . --python 3.12 --reinstall`。
- **跳过一次同步**：`SEMSEARCH_SKIP_SYNC=1 git commit ...`。
- **发布版本**：一键安装命令下载的是最新 GitHub release 附带的安装脚本。用 `sh scripts/release-assets.sh v1.2.3 dist/release` 生成这两个文件，其中安装的版本固定为该标签（仓库里的脚本装的是 `main`）；再用 `gh release create v1.2.3 dist/release/install.ps1 dist/release/install.sh` 附到 release 上。

评测：
```bash
uv run python evals/index_projects.py <项目根目录> ...
uv run python evals/run.py --modes "vector,hybrid,hybrid;tr=1" --show-misses
```
- 题目放在 `evals/cases/`，格式参考 `semsearch.json`。本机找不到对应项目目录的题目集会被跳过；`raftmod.json` 需要本地有一份 [RaftLocalCoop](https://github.com/spardanviro/RaftLocalCoop)。
- 私人项目的题目放 `evals/private/cases/`：这个目录不会被提交，评测时会自动读取。

## 许可证
本项目代码采用 [MIT](LICENSE) 许可证。EmbeddingGemma 2 模型不包含在仓库中，需要单独下载；模型的使用条款以 Google 在 Hugging Face 上发布的为准，不受 MIT 许可证约束。
