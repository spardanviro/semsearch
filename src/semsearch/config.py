"""Settings and defaults. Every value can be overridden with a SEMSEARCH_* environment variable."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"
EMBED_DIM = 768
CHUNKER_VERSION = "4"   # 2: "\n"-only lines, no boilerplate gaps; 3: lead trim;
                        # 4: doc comments join their definition, PowerShell functions
SCHEMA_VERSION = "3"    # 2: AUTOINCREMENT chunk ids; 3: chunks_fts keyword index
KEYWORDS_VERSION = "1"  # bump when keywords.terms() changes: the FTS index is rebuilt
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
SENSITIVE_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*", "id_ecdsa*",
    "*.keystore", "*.jks", "secrets.*", "*secret*.json", "*secret*.yaml", "*secret*.yml",
    "credentials*", ".npmrc", ".pypirc", ".netrc", "_netrc", ".git-credentials", ".pgpass",
    "*.tfvars", "*.tfstate", "*.tfstate.*", "kubeconfig", "*.kubeconfig", ".docker/config.json",
    "appsettings.*.json", "*.token", "*.kdbx", "*.ovpn", "service-account*.json",
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


def default_home() -> Path:
    return Path(os.environ.get("SEMSEARCH_HOME", Path.home() / ".cache" / "semsearch"))


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
    target_tokens: int = 600                 # preferred chunk size
    max_tokens: int = 1024                   # hard limit (model max_seq_length)
    window_lines: int = 60
    window_overlap: int = 15
    batch_token_budget: int = field(
        default_factory=lambda: _env_int("SEMSEARCH_BATCH_TOKENS", 16 * 1024))
    commit_every_chunks: int = 256
    scan_cooldown_s: float = 5.0
    search_wait_s: float = 1.5
    preview_chars: int = 600
    per_file_results: int = 2
    search_mode: str = field(default_factory=lambda: os.environ.get("SEMSEARCH_SEARCH_MODE", "hybrid"))
    keyword_weight: float = field(
        default_factory=lambda: _env_float("SEMSEARCH_KEYWORD_WEIGHT", DEFAULT_KEYWORD_WEIGHT))

    def __post_init__(self) -> None:
        if self.search_mode not in SEARCH_MODES:
            raise ValueError(f"SEMSEARCH_SEARCH_MODE must be one of {SEARCH_MODES}, "
                             f"got {self.search_mode!r}")
        if not (math.isfinite(self.keyword_weight) and self.keyword_weight >= 0):
            raise ValueError(f"SEMSEARCH_KEYWORD_WEIGHT must be a finite number >= 0, "
                             f"got {self.keyword_weight!r}")

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
