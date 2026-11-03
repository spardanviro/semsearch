# semsearch

[中文说明](README.zh-CN.md)

Local semantic code search for [Claude Code](https://claude.com/claude-code), powered by
[EmbeddingGemma 2](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/).

Claude Code finds code with Grep and LSP, which works well when you know the name. semsearch is
for when you only know **what the code does**. It finds that code by meaning: "where are failed
downloads retried", "the code that keeps a crash from corrupting the save file", "anything
similar to this snippet". It adds to Grep and LSP and replaces neither.

- **Runs locally.** Everything stays on your machine: one background daemon holds the model,
  and the indexes live in `~/.cache/semsearch`.
- **Updates incrementally.** Only changed files are re-chunked, and only changed chunks are
  re-embedded. A hook updates the index after each Claude edit, and a quick scan before each
  search catches everything else (editor saves, `git pull`, branch switches).
- **Splits code by syntax.** tree-sitter cuts code into functions, methods and classes, and a
  doc comment stays in the same chunk as the code it describes. Markdown is split by heading.
- **Built into Claude Code.** It ships an MCP tool, a `semantic-search` subagent, an
  edit hook and routing rules, so Claude picks the right tool by itself.
- **Images, PDFs, audio and video (opt-in).** These are searchable by meaning too: "the blue
  explosion sprite sheet", "the page with the voxel visibility diagram", "the clip where the
  character goes limp". See [Media](#media).

## How well it works

| Benchmark | Questions | MRR | Answer in top 5 |
|---|---|---|---|
| 6 private game-modding projects (C#, TS, GDScript, PowerShell, JS; up to 60k chunks) | 72 | 0.83 | 96% |
| A public C# mod and this repository (`evals/cases/`) | 54 | 0.79 | 91% |

- **Who wrote the questions:** for the 72-question set, agents that had not seen the
  implementation, with every expected answer checked against the source. Those cases stay on
  the author's machine because the projects are private. The 54-question set is in this repo;
  it was written by the developer, so it may be easier than real use. Its questions about this
  repository drift as the code changes, so their expected answers are updated when code
  moves (see [docs/AUDIT.md](docs/AUDIT.md)).
- **How they were asked:** the way Claude asks in practice: a non-English question is
  translated to English first. The scores are for the hybrid mode; pure vector, the default,
  is within 0.01 MRR on the 72-question set and 0.03 lower on the 54-question set.
- **Details:** the full methodology and every experiment are in
  [docs/EVAL_REAL_PROJECTS.md](docs/EVAL_REAL_PROJECTS.md) and
  [docs/PHASE1_5.md](docs/PHASE1_5.md). Those docs are in Chinese.

Speed (RTX 3060 6 GB):
- **Warm query:** about 0.7 s, including CLI start-up.
- **First query after the daemon starts:** about 20 s, while the model loads.
- **First index:** about 1.5 min for 250 files; about 26 min for 8,000 files / 60k chunks.

## Requirements

- **Python 3.12** and [uv](https://docs.astral.sh/uv/).
- **An NVIDIA GPU is recommended.** It is not required: without one the model runs on the CPU,
  just much more slowly.
- **PyTorch build:** on Windows and Linux, `pyproject.toml` installs the CUDA 13.0 build of
  PyTorch (`pytorch-cu130`), which needs NVIDIA driver 580 or newer. With an older driver the
  model silently falls back to the CPU (`semsearch doctor` detects this). For another CUDA version, or for CPU only, change
  `[tool.uv.sources]` / `[[tool.uv.index]]`. macOS uses the regular PyPI build.
- **Platform:** developed and tested on Windows 11. The code is cross-platform, but Linux and
  macOS have not been tested yet.
- **Model download:** the model, `google/embeddinggemma-2` (1.5 GB), is downloaded from
  Hugging Face on first use, or ahead of time with `semsearch download-model`, which asks first.

## Install

### One-line install (recommended)

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/spardanviro/SearchEngine/main/scripts/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/spardanviro/SearchEngine/main/scripts/install.sh | sh
```

The script installs [uv](https://docs.astral.sh/uv/) if it is missing, installs semsearch with
every dependency pinned to `uv.lock`, runs `semsearch doctor`, offers to download the model,
and finally runs `semsearch install-claude`. Each download and each change to your setup asks
first. It needs about 5 GB of disk (PyTorch about 3.5 GB, the model 1.5 GB). The Windows script
has been tested on Windows 11; `install.sh` so far only under Git Bash on Windows.

Options, as environment variables: `SEMSEARCH_INSTALL_REF` (branch, tag or commit; default
`main`), `SEMSEARCH_INSTALL_SOURCE` (a local folder to install from) and
`SEMSEARCH_INSTALL_YES=1` (answer yes to every question).

### Manual install

```bash
git clone https://github.com/spardanviro/SearchEngine.git
cd SearchEngine
uv tool install . --python 3.12
semsearch install-claude --dry-run   # show what would change
semsearch install-claude             # register with Claude Code (asks first)
```

`uv tool install` copies semsearch into its own environment, so the clone can be moved or
deleted afterwards.

`install-claude` changes only user-level Claude Code files, and it backs each one up first
(`*.semsearch.bak`):

| File | Change |
|---|---|
| `~/.claude.json` | adds the `semsearch` MCP server |
| `~/.claude/settings.json` | adds a PostToolUse hook that tells the daemon about edited files |
| `~/.claude/agents/semantic-search.md` | the subagent definition |
| `~/.claude/CLAUDE.md` | a marked block with tool routing rules (it replaces an existing `# 代码导航` section) |

- The routing rules are written in Chinese. Edit
  `src/semsearch/integrations/claude-md-snippet.md` before installing if you prefer English.
- Every touched file keeps two backups: `*.semsearch.bak` is the state before semsearch first
  touched it, and `*.semsearch.prev` is the state before the latest change. They are full
  copies: `~/.claude.json.semsearch.bak` contains whatever `~/.claude.json` held, tokens
  included.

Open a new Claude Code session afterwards so it loads the tools. To remove everything:

```bash
semsearch install-claude --uninstall
uv tool uninstall semsearch
```

## Usage

### In Claude Code

Usually you don't do anything. Ask questions as you normally would, and Claude chooses the tool:

| You know… | Claude uses |
|---|---|
| the exact name, string or regex | Grep / Glob |
| a symbol, and want its definition, references or callers | LSP |
| only what the code does | `semantic_search` |
| an open question that needs several searches plus reading | the `semantic-search` subagent |

To ask for it explicitly, say "use semantic search to find …" or "ask the semantic-search
subagent where …".

The first search in a project builds its index in the background. Until the index is ready,
results are marked as incomplete and Claude falls back to Grep.

### Command line

```bash
semsearch search "how the second player respawns after being downed" -k 5
semsearch search "save file backup" -g "src/**" -m code --json
semsearch status
semsearch reindex            # rescan now
semsearch reindex --full     # rebuild the index from scratch
semsearch daemon stop        # free GPU memory; the next search restarts it
semsearch doctor             # check Python, GPU and driver, model, disk, Claude Code setup
semsearch download-model     # download the model now (shows the size and asks first)
```

`semsearch doctor` prints a fix for every problem it finds (for example "NVIDIA driver too old,
running on the CPU" or "Hugging Face cannot be reached"). It contacts Hugging Face only when the
model is not downloaded yet; `--offline` skips that.

`--root <dir>` searches a project other than the current one.

### Excluding files

- **`.gitignore`:** anything it excludes is never indexed.
- **`.semsearchignore`:** add one at the project root, using the same syntax, to exclude more.
- **Secrets:** files such as `.env*`, `*.pem`, `*.key`, `id_rsa*` and credentials files are
  always skipped, whatever their case or folder, even if an ignore rule tries to re-include
  them. See `SENSITIVE_PATTERNS` in `src/semsearch/config.py`.
- **Generated files:** minified bundles are skipped, as are files whose lines average more
  than 500 characters of content.
- **Folders that are never indexed:** the home directory itself, a drive root, `~/.claude`
  and `~/.ssh`. Open Claude Code in a project folder.

## Media

EmbeddingGemma 2 maps text, images, audio and video into one vector space, so a text question
can find any of them. Media indexing is **off by default**. To turn it on:

```bash
setx SEMSEARCH_MEDIA "image,pdf"     # or: image,pdf,audio,video  /  all
semsearch daemon stop
```

| Kind | One chunk is | Shown as |
|---|---|---|
| image (png, jpg, webp, bmp, gif) | the picture itself (icons under 32×32 are skipped) | `art/hero.png` |
| pdf | one page: its text plus the rendered page | `docs/manual.pdf p.12` |
| audio (wav, mp3, ogg, flac, m4a, …) | a 30 s window, with 5 s of overlap | `sfx/theme.ogg 01:05-01:35` |
| video (mp4, webm, mkv, mov, …) | a 20 s segment at 1 frame per second, plus its sound if audio is enabled | `clips/intro.mp4 00:40-01:00` |

- Only the encoders the enabled kinds need are loaded (all of them take about 1.4 GB of VRAM).
  Turning media on, or changing the list, rebuilds each index once.
- Claude can open image and PDF hits with Read (a PDF hit names its page). Audio and video
  hits give a time range.
- Benchmark (30 questions over Godot documentation images and clips plus a paper):
  - **Images:** MRR 0.94, with the answer in the top 5 every time.
  - **PDF pages and video segments:** the answer is in the top 5 for 83% of questions.
  - Details: [docs/PHASE2_3.md](docs/PHASE2_3.md).
- Game projects can hold thousands of textures. Exclude asset folders you never search with
  `.semsearchignore`.
- Limits: `SEMSEARCH_MAX_MEDIA_BYTES` (default 500 MB per file), `SEMSEARCH_MAX_PDF_PAGES`
  (default 500) and `SEMSEARCH_MAX_MEDIA_SECONDS` (default 3600; longer recordings are indexed up
  to that point).

## Configuration

Settings come from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SEMSEARCH_HOME` | `~/.cache/semsearch` | indexes, logs, daemon state |
| `SEMSEARCH_IDLE_TIMEOUT` | `1200` | seconds of inactivity before the daemon exits |
| `SEMSEARCH_SEARCH_MODE` | `vector` | `vector`, `hybrid` (vector plus a BM25 bonus for identifier-like queries) or `keyword` |
| `SEMSEARCH_KEYWORD_WEIGHT` | `0.05` | the maximum BM25 bonus in `hybrid` mode |
| `SEMSEARCH_MAX_FILE_BYTES` | `1000000` | larger source/doc files are skipped (config files such as json/yaml/xml: fixed 200 KB) |
| `SEMSEARCH_MAX_FILES` | `50000` | the per-project file limit |
| `SEMSEARCH_BATCH_TOKENS` | `16384` | the embedding batch token budget; lower it if the GPU runs out of memory |
| `SEMSEARCH_MODEL` | `google/embeddinggemma-2` | any sentence-transformers model id or local path |
| `SEMSEARCH_EMBED_DIM` | `0` | vector size; `0` uses the model's own size, a smaller value truncates (for Matryoshka models) |
| `SEMSEARCH_QUERY_PROMPT` | automatic | a prompt name from the model card, or literal prefix text (`""` = none) |
| `SEMSEARCH_DOCUMENT_PROMPT` | automatic | the same, for indexed chunks |
| `SEMSEARCH_CHUNK_TOKENS` | `600` | preferred chunk size in tokens |
| `SEMSEARCH_MAX_TOKENS` | `1024` | hard chunk limit; it is capped at the model's own limit, with a warning |
| `SEMSEARCH_MODEL_REVISION` | pinned commit | Hugging Face revision; the default model is pinned to the tested commit, other models use their default branch |
| `SEMSEARCH_TRUST_REMOTE_CODE` | `false` | allow the model repository to run its own Python code |
| `SEMSEARCH_MEDIA` | *(off)* | media kinds to index: `image`, `pdf`, `audio`, `video`, or `all` |
| `SEMSEARCH_MAX_MEDIA_BYTES` | `500000000` | larger media files are skipped |
| `SEMSEARCH_MAX_PDF_PAGES` | `500` | pages indexed per PDF |
| `SEMSEARCH_MAX_MEDIA_SECONDS` | `3600` | how much of each audio/video file is indexed |
| `SEMSEARCH_EMBEDDER` | `st` | embedding backend: `st` (sentence-transformers) or `fake` (tests only) |

When the two prompt variables are unset, the choice is automatic:
- EmbeddingGemma uses its `SearchQuery` / `Document` prompts.
- Another model uses the `query` / `document` prompts from its own sentence-transformers
  config, if it defines them.
- Otherwise no prompt is used.

The daemon reads these settings when it starts. After changing one, run
`semsearch daemon stop`; the next search starts the daemon again with the new values. Set them
as user environment variables so that the daemon sees them no matter whether Claude Code or
the CLI started it.

### Switching the model

Any embedding model that sentence-transformers can load works without code changes. For
example:

```bash
# Windows: setx SEMSEARCH_MODEL "Qwen/Qwen3-Embedding-0.6B"  (then open a new terminal)
export SEMSEARCH_MODEL="Qwen/Qwen3-Embedding-0.6B"   # ships its own query/document prompts
semsearch daemon stop
```

- **Prompts matter.** Every model expects its own query and document prefixes, and the wrong
  ones noticeably hurt ranking. If the model's sentence-transformers config does not define
  `query` / `document` prompts, take them from the model card and set
  `SEMSEARCH_QUERY_PROMPT` / `SEMSEARCH_DOCUMENT_PROMPT`. For example, E5 models use
  `"query: "` and `"passage: "`. A name the model does not know is used as literal text, and
  the daemon logs a warning.
- **Context length.** If the model handles fewer than 1024 tokens, lower `SEMSEARCH_MAX_TOKENS`
  and `SEMSEARCH_CHUNK_TOKENS` to match. Otherwise the model's own limit applies, and long chunks
  lose their tail.
- **Indexes rebuild automatically.** The model id, vector size, document prompt and chunk sizes
  are recorded with each index, and a change triggers a rebuild. Changing only the query prompt
  does not.
- **`SEMSEARCH_TRUST_REMOTE_CODE`.** Some models need it. It runs Python code downloaded from
  the model repository, so turn it on only for publishers you trust.
- **Measure it.** Compare models on your own code with the benchmark before switching. Give
  each model its own index folder so they don't overwrite each other:
  `uv run python evals/run.py --home evals/.home-<model> ...`.
- **Other backends.** Ollama or a cloud API can be added by implementing the small `Embedder`
  protocol in `src/semsearch/embed/base.py` and registering it in `make_embedder`
  (`SEMSEARCH_EMBEDDER`). A cloud API sends your code off the machine.

## How it works

```
Claude Code ── MCP (stdio) ──┐
semsearch CLI ───────────────┼── HTTP on 127.0.0.1 + token ──> semsearchd
PostToolUse hook ────────────┘                                 ├─ EmbeddingGemma 2 (one copy)
                                                               ├─ per-project SQLite index
                                                               │   (chunks, vectors, FTS5)
                                                               └─ in-memory vector matrix
```

- **One daemon** serves every project. It starts on demand and exits when idle.
- **Storage:** SQLite (WAL) is the source of truth, and vectors are stored as BLOBs. Search is
  a brute-force matrix product in numpy, which was about 30× faster than sqlite-vec at this
  scale.
- **Change detection:** files are compared first by mtime and size, then by a blake3 content
  hash. Chunks are matched by hash, so moved or copied code reuses its existing vectors.
- **Duplicate code:** identical code in copied folders shows up as a single result that lists
  the other copies.
- **Untrusted content:** paths, symbol names and previews come from repository files. Before
  Claude sees them, invisible characters are stripped, length is capped, and they are labeled
  as untrusted.
- **Private data:** the daemon listens on 127.0.0.1 only and requires a token. Its folder (the
  token, indexes and logs) is made readable by the current user only; on Windows this is done
  with an explicit ACL. Request bodies are capped at 1 MB and requests are rate-limited.
- **Media files:** images, PDFs, audio and video are decoded in a separate worker process with
  a memory cap and a time limit (on Windows it may not start other processes). A file that
  crashes or hangs it only loses its own chunks. The worker runs as your user, so it is a
  damage limit, not a full sandbox.

The full design (in Chinese) is in [docs/DESIGN.md](docs/DESIGN.md).

## Development

```bash
uv sync
git config core.hooksPath .githooks   # after each commit, refresh the installed tool
uv run pytest -q
uv run ruff check src tests evals
uv run pyright src tests evals
```

- **Real-model tests:** they are skipped by default. Set `SEMSEARCH_GPU_TESTS=1` to run them
  (`uv run pytest -q -m gpu`).
- **Test isolation:** the tests ignore your own `SEMSEARCH_*` variables.
- **Post-commit hook:** when a commit touches `src/`, `pyproject.toml` or `uv.lock`, it
  reinstalls the uv tool copy from a snapshot of `HEAD`, with dependencies pinned to
  `uv.lock`. Uncommitted edits never reach Claude Code. The hook does not run on pull, checkout
  or rebase; run `uv tool install . --python 3.12 --reinstall` after those. Skip it once with
  `SEMSEARCH_SKIP_SYNC=1 git commit ...`.

Benchmarks:

```bash
uv run python evals/index_projects.py <project root> ...
uv run python evals/run.py --modes "vector,hybrid,hybrid;tr=1" --show-misses
```

- Case files live in `evals/cases/`; see `semsearch.json` for the format. A suite whose
  project folder does not exist on your machine is skipped; `raftmod.json` needs a local
  checkout of [RaftLocalCoop](https://github.com/spardanviro/RaftLocalCoop).
- Put cases about private projects in `evals/private/cases/`. That folder is gitignored and is
  picked up automatically.

## License

[MIT](LICENSE). EmbeddingGemma 2 is downloaded separately. Its use is governed by Google's own
model terms on Hugging Face, not by this license.
