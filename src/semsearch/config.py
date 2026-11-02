"""Settings and defaults. Every value can be overridden with a SEMSEARCH_* environment variable."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"   # default model (SEMSEARCH_MODEL)
# pinned Hugging Face commit of the default model: what the benchmarks ran on; a change of the
# repo's default branch must not silently swap the weights (SEMSEARCH_MODEL_REVISION)
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
EMBED_DIM = 768                        # its native vector size
MIN_MAX_TOKENS = 64
CHUNKER_VERSION = "5"   # 2: "\n"-only lines, no boilerplate gaps; 3: lead trim;
                        # 4: doc comments join their definition, PowerShell functions;
                        # 5: long lines split, short text gaps kept, JS arrow functions,
                        #    Go/Rust names, Rust attributes, minified files skipped
SCHEMA_VERSION = "3"    # 2: AUTOINCREMENT chunk ids; 3: chunks_fts keyword index
KEYWORDS_VERSION = "2"  # bump when keywords.terms() changes: the FTS index is rebuilt
SEARCH_MODES = ("hybrid", "vector", "keyword")
DEFAULT_KEYWORD_WEIGHT = 0.05   # max BM25 bonus added to cosine (tuned on evals/, docs/PHASE1_5.md)

# file extension -> tree-sitter language name
CODE_LANGS: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".cs": "csharp",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".c": "c", ".h": "cpp", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".hh": "cpp", ".hxx": "cpp",
    ".gd": "gdscript", ".lua": "lua", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".scala": "scala", ".dart": "dart", ".zig": "zig",
    ".sh": "bash", ".bash": "bash", ".ps1": "powershell",
    ".sql": "sql", ".shader": "hlsl", ".hlsl": "hlsl", ".glsl": "glsl", ".cginc": "hlsl",
    ".vue": "vue", ".svelte": "svelte",
}
DOC_EXTS = {".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text", ".adoc": "text"}
CONFIG_EXTS = {
    ".json": "json", ".jsonc": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".xml": "xml", ".gradle": "groovy", ".cmake": "cmake",
}
SPECIAL_FILENAMES = {"dockerfile": "dockerfile", "makefile": "make", "cmakelists.txt": "cmake"}
# media (phase 2/3): indexed only for kinds listed in SEMSEARCH_MEDIA
MEDIA_KINDS = ("image", "pdf", "audio", "video")
MEDIA_EXTS = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".bmp": "image",
    ".gif": "image",
    ".pdf": "pdf",
    ".wav": "audio", ".mp3": "audio", ".ogg": "audio", ".flac": "audio", ".m4a": "audio",
    ".aac": "audio", ".opus": "audio",
    ".mp4": "video", ".webm": "video", ".mkv": "video", ".mov": "video", ".avi": "video",
    ".m4v": "video",
}
MEDIA_VERSION = "1"     # bump when media segmentation/rendering changes: media chunks re-embed

# never source, at any depth (compared case-insensitively)
ALWAYS_EXCLUDE_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", ".idea", ".vs", ".gradle", ".next", ".nuxt",
    ".cache", ".godot", ".import",
})
# usually build output - but sometimes real source ("src/build/", "bin/" scripts). Only applied
# when the project has no .gitignore to tell us what is generated.
BUILD_EXCLUDE_DIRS = frozenset({
    "dist", "build", "target", "bin", "obj", "coverage",
    "library", "temp", "logs", "usersettings",          # Unity
})
# matched case-insensitively against project-relative paths, and enforced even when a
# repository's ignore files try to re-include them
SENSITIVE_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*", "id_ecdsa*",
    "*.keystore", "*.jks", "secrets.*", "*secret*.json", "*secret*.yaml", "*secret*.yml",
    "*secret*.txt", "credentials*", "*credentials*.json", "*credentials*.yaml",
    "*credentials*.yml", "passwords*", "*password*.txt", "token*.json", "auth.json",
    ".npmrc", ".pypirc", ".netrc", "_netrc", ".git-credentials", ".pgpass",
    "*.tfvars", "*.tfvars.json", "*.tfstate", "*.tfstate.*", "kubeconfig", "*.kubeconfig",
    "**/.docker/config.json", "**/gh/hosts.yml", "appsettings.*.json", "local.settings.json",
    "*adminsdk*.json", "service-account*.json", "serviceaccount*.json", "wp-config.php",
    "*.token", "*.kdbx", "*.ovpn",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name} must be true/false, got {raw!r}")


def _env_media(name: str) -> frozenset[str]:
    """`image,pdf,audio,video`, `all`, or empty/`none`."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("", "none", "off", "0"):
        return frozenset()
    if raw == "all":
        return frozenset(MEDIA_KINDS)
    kinds = frozenset(k.strip() for k in raw.split(",") if k.strip())
    unknown = kinds - set(MEDIA_KINDS)
    if unknown:
        raise ValueError(f"{name}: unknown media kinds {sorted(unknown)}; use any of "
                         f"{', '.join(MEDIA_KINDS)} or 'all'")
    return kinds


def default_home() -> Path:
    raw = os.environ.get("SEMSEARCH_HOME", "").strip()   # "" would mean the cwd
    return Path(raw).expanduser() if raw else Path.home() / ".cache" / "semsearch"


@dataclass(frozen=True)
class Settings:
    home: Path = field(default_factory=default_home)
    embedder: str = field(default_factory=lambda: os.environ.get("SEMSEARCH_EMBEDDER", "st"))
    idle_timeout_s: float = field(
        default_factory=lambda: _env_float("SEMSEARCH_IDLE_TIMEOUT", 20 * 60))
    max_file_bytes: int = field(
        default_factory=lambda: _env_int("SEMSEARCH_MAX_FILE_BYTES", 1_000_000))
    max_config_bytes: int = 200_000          # json/yaml/xml larger than this are usually data
    max_files: int = field(default_factory=lambda: _env_int("SEMSEARCH_MAX_FILES", 50_000))
    # --- embedding model (docs: README "Switching the model") ---------------------------------
    model: str = field(default_factory=lambda: os.environ.get("SEMSEARCH_MODEL", MODEL_ID))
    # None = the pinned commit for the default model, the repo's default branch otherwise
    model_revision: str | None = field(
        default_factory=lambda: os.environ.get("SEMSEARCH_MODEL_REVISION") or None)
    embed_dim: int = field(                  # 0 = the model's own size; smaller = truncate
        default_factory=lambda: _env_int("SEMSEARCH_EMBED_DIM", 0))
    # model-card prompt name or literal prefix text; None (unset) = automatic per model
    query_prompt: str | None = field(
        default_factory=lambda: os.environ.get("SEMSEARCH_QUERY_PROMPT"))
    document_prompt: str | None = field(
        default_factory=lambda: os.environ.get("SEMSEARCH_DOCUMENT_PROMPT"))
    trust_remote_code: bool = field(         # lets the model repo run its own Python code
        default_factory=lambda: _env_bool("SEMSEARCH_TRUST_REMOTE_CODE", False))
    target_tokens: int = field(              # preferred chunk size
        default_factory=lambda: _env_int("SEMSEARCH_CHUNK_TOKENS", 600))
    max_tokens: int = field(                 # hard limit = the model's max_seq_length
        default_factory=lambda: _env_int("SEMSEARCH_MAX_TOKENS", 1024))
    # --- media (images, PDF pages, audio, video): off unless listed ---------------------------
    media: frozenset[str] = field(default_factory=lambda: _env_media("SEMSEARCH_MEDIA"))
    max_media_bytes: int = field(
        default_factory=lambda: _env_int("SEMSEARCH_MAX_MEDIA_BYTES", 500_000_000))
    max_pdf_pages: int = field(default_factory=lambda: _env_int("SEMSEARCH_MAX_PDF_PAGES", 500))
    max_media_seconds: float = field(        # audio/video beyond this are indexed up to it
        default_factory=lambda: _env_float("SEMSEARCH_MAX_MEDIA_SECONDS", 3600.0))
    window_lines: int = 60
    window_overlap: int = 15
    batch_token_budget: int = field(
        default_factory=lambda: _env_int("SEMSEARCH_BATCH_TOKENS", 16 * 1024))
    commit_every_chunks: int = 256
    scan_cooldown_s: float = 5.0
    search_wait_s: float = 1.5
    preview_chars: int = 600
    per_file_results: int = 2
    # vector by default: hybrid showed no gain over 126 eval cases (docs/EVAL_REAL_PROJECTS.md)
    search_mode: str = field(default_factory=lambda: os.environ.get("SEMSEARCH_SEARCH_MODE", "vector"))
    keyword_weight: float = field(
        default_factory=lambda: _env_float("SEMSEARCH_KEYWORD_WEIGHT", DEFAULT_KEYWORD_WEIGHT))

    def __post_init__(self) -> None:
        if self.search_mode not in SEARCH_MODES:
            raise ValueError(f"SEMSEARCH_SEARCH_MODE must be one of {SEARCH_MODES}, "
                             f"got {self.search_mode!r}")
        if not (math.isfinite(self.keyword_weight) and self.keyword_weight >= 0):
            raise ValueError(f"SEMSEARCH_KEYWORD_WEIGHT must be a finite number >= 0, "
                             f"got {self.keyword_weight!r}")
        if not self.model.strip() or self.model != self.model.strip():
            raise ValueError(f"SEMSEARCH_MODEL must be a non-empty name, got {self.model!r}")
        positive = {"SEMSEARCH_MAX_FILES": self.max_files,
                    "SEMSEARCH_MAX_FILE_BYTES": self.max_file_bytes,
                    "SEMSEARCH_BATCH_TOKENS": self.batch_token_budget,
                    "SEMSEARCH_IDLE_TIMEOUT": self.idle_timeout_s}
        for name, value in positive.items():
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        if self.embed_dim < 0:
            raise ValueError(f"SEMSEARCH_EMBED_DIM must be >= 0, got {self.embed_dim}")
        # checked here, before any index meta is compared: a bad value must not wipe an index
        if self.model == MODEL_ID and self.embed_dim > EMBED_DIM:
            raise ValueError(f"SEMSEARCH_EMBED_DIM={self.embed_dim} exceeds {MODEL_ID}'s "
                             f"vector size {EMBED_DIM}")
        if self.max_tokens < MIN_MAX_TOKENS:
            raise ValueError(f"SEMSEARCH_MAX_TOKENS must be >= {MIN_MAX_TOKENS}, "
                             f"got {self.max_tokens}")
        if self.media and "embeddinggemma" not in self.model.lower():
            raise ValueError("SEMSEARCH_MEDIA needs a multimodal EmbeddingGemma model "
                             f"(SEMSEARCH_MODEL={self.model!r})")
        if self.max_pdf_pages <= 0 or self.max_media_seconds <= 0 or self.max_media_bytes <= 0:
            raise ValueError("SEMSEARCH_MAX_PDF_PAGES / MAX_MEDIA_SECONDS / MAX_MEDIA_BYTES "
                             "must be > 0")
        if not 0 < self.target_tokens <= self.max_tokens:
            raise ValueError(f"need 0 < SEMSEARCH_CHUNK_TOKENS ({self.target_tokens}) <= "
                             f"SEMSEARCH_MAX_TOKENS ({self.max_tokens})")

    @property
    def revision(self) -> str | None:
        if self.model_revision:
            return self.model_revision
        return MODEL_REVISION if self.model == MODEL_ID else None

    @property
    def projects_dir(self) -> Path:
        return self.home / "projects"

    @property
    def daemon_file(self) -> Path:
        return self.home / "daemon.json"

    @property
    def lock_file(self) -> Path:
        return self.home / "daemon.lock"

    @property
    def log_dir(self) -> Path:
        return self.home / "logs"
