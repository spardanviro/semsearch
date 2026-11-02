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

## How well it works

All numbers come from benchmarks in this repo (`evals/`). The questions were written by agents
that had not seen the implementation, and every expected answer was checked against the source.

| Benchmark | Questions | MRR | Answer in top 5 |
|---|---|---|---|
| 6 real game-modding projects (C#, TS, GDScript, PowerShell, JS; up to 60k chunks) | 72 | 0.83 | 96% |
| A public C# mod and this repository | 54 | 0.76 | 87% |

These scores use the setup Claude runs in practice: a non-English question is first
translated to English. The full methodology and every experiment we tried are in
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
- **PyTorch build:** `pyproject.toml` installs the CUDA 13.0 build of PyTorch
  (`pytorch-cu130`). For a different CUDA version, or for CPU only, change
  `[tool.uv.sources]` / `[[tool.uv.index]]`.
- **Platform:** developed and tested on Windows 11. The code is cross-platform, but Linux and
  macOS have not been tested yet.
- **Model download:** the model, `google/embeddinggemma-2`, is downloaded from Hugging Face on
  first use.

## Install

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
| `~/.claude/CLAUDE.md` | a marked block with tool routing rules |

The routing rules are written in Chinese. Edit
`src/semsearch/integrations/claude-md-snippet.md` before installing if you prefer English.

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
```

`--root <dir>` searches a project other than the current one.

### Excluding files

- **`.gitignore`:** anything it excludes is never indexed.
- **`.semsearchignore`:** add one at the project root, using the same syntax, to exclude more.
- **Secrets:** files such as `.env*`, `*.pem`, `*.key`, `id_rsa*` and credentials files are
  always skipped, even if an ignore rule tries to re-include them.

## Configuration

Settings come from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SEMSEARCH_HOME` | `~/.cache/semsearch` | indexes, logs, daemon state |
| `SEMSEARCH_IDLE_TIMEOUT` | `1200` | seconds of inactivity before the daemon exits |
| `SEMSEARCH_SEARCH_MODE` | `vector` | `vector`, `hybrid` (vector plus a BM25 bonus for identifier-like queries) or `keyword` |
| `SEMSEARCH_KEYWORD_WEIGHT` | `0.05` | the maximum BM25 bonus in `hybrid` mode |
| `SEMSEARCH_MAX_FILE_BYTES` | `1000000` | larger files are skipped |
| `SEMSEARCH_MAX_FILES` | `50000` | the per-project file limit |
| `SEMSEARCH_BATCH_TOKENS` | `16384` | the embedding batch token budget; lower it if the GPU runs out of memory |
| `SEMSEARCH_MODEL` | `google/embeddinggemma-2` | any sentence-transformers model id or local path |
| `SEMSEARCH_EMBED_DIM` | `0` | vector size; `0` uses the model's own size, a smaller value truncates (for Matryoshka models) |
| `SEMSEARCH_QUERY_PROMPT` | automatic | a prompt name from the model card, or literal prefix text (`""` = none) |
| `SEMSEARCH_DOCUMENT_PROMPT` | automatic | the same, for indexed chunks |
| `SEMSEARCH_CHUNK_TOKENS` | `600` | preferred chunk size in tokens |
| `SEMSEARCH_MAX_TOKENS` | `1024` | hard chunk limit; it is capped at the model's own limit, with a warning |

When the two prompt variables are unset, the choice is automatic:
- EmbeddingGemma uses its `SearchQuery` / `Document` prompts.
- Another model uses the `query` / `document` prompts from its own sentence-transformers
  config, if it defines them.
- Otherwise no prompt is used.
| `SEMSEARCH_TRUST_REMOTE_CODE` | `false` | allow the model repository to run its own Python code |

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
- **Untrusted previews:** previews are sanitized and labeled as untrusted file content before
  Claude sees them.

The full design (in Chinese) is in [docs/DESIGN.md](docs/DESIGN.md).

## Development

```bash
uv sync
git config core.hooksPath .githooks   # after each commit, refresh the installed tool
uv run pytest -q
uv run ruff check src tests evals
pyright src tests evals
```

The post-commit hook reinstalls the uv tool copy whenever a commit touches `src/`,
`pyproject.toml` or `uv.lock`, so Claude Code always runs committed code.

Benchmarks:

```bash
uv run python evals/index_projects.py <project root> ...
uv run python evals/run.py --modes "vector,hybrid,hybrid;tr=1" --show-misses
```

- Case files live in `evals/cases/`; see `semsearch.json` for the format.
- Put cases about private projects in `evals/private/cases/`. That folder is gitignored and is
  picked up automatically.

## Roadmap

- Images and PDFs (EmbeddingGemma 2 is multimodal)
- Audio and video

## License

[MIT](LICENSE). EmbeddingGemma 2 is downloaded separately. Its use is governed by Google's own
model terms on Hugging Face, not by this license.
