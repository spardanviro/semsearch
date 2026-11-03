#!/bin/sh
# semsearch installer for macOS and Linux.
#
#   curl -LsSf https://github.com/spardanviro/semsearch/releases/latest/download/install.sh | sh
#
# or, from a clone:  sh scripts/install.sh
#
# Steps; each one that downloads something or changes your setup asks first:
#   1. uv (Python tool manager) from astral.sh, if missing
#   2. semsearch with dependencies pinned to uv.lock (PyTorch: about 3.5 GB on Linux)
#   3. `semsearch doctor`: checks GPU, driver, model, disk, Claude Code
#   4. the embedding model, about 1.5 GB (`semsearch download-model`)
#   5. registration with Claude Code (`semsearch install-claude`)
#
# Environment: SEMSEARCH_INSTALL_REF     branch, tag or commit to install (default: DEFAULT_REF below)
#              SEMSEARCH_INSTALL_SOURCE  install from this local folder instead of GitHub
#              SEMSEARCH_INSTALL_YES=1   answer yes to every question
set -eu

REPO=spardanviro/semsearch
DEFAULT_REF=main   # DEFAULT_REF: a release's copy of this script names its own tag
REF=${SEMSEARCH_INSTALL_REF:-$DEFAULT_REF}
YES=${SEMSEARCH_INSTALL_YES:-}

step() { printf '\n==> %s\n' "$1"; }
note() { printf '    %s\n' "$1"; }
fail() {
    printf '\nInstallation failed: %s\n' "$1" >&2
    printf 'Claude Code is registered only in the last step, so its settings were not changed.\n' >&2
    exit 1
}

# `curl ... | sh` feeds this script on stdin: questions read the terminal instead, and child
# commands get </dev/null so they can never swallow the rest of the script
if (exec </dev/tty) 2>/dev/null; then TTY=/dev/tty; else TTY=/dev/null; fi

confirm() {
    [ "$YES" = 1 ] && return 0
    [ "$TTY" = /dev/tty ] || return 1
    printf '%s [y/N] ' "$1" >/dev/tty
    read -r answer </dev/tty || return 1
    [ "$answer" = y ] || [ "$answer" = Y ]
}

is_source() {   # a semsearch checkout, not just any folder with a uv.lock
    [ -f "$1/uv.lock" ] && grep -qx 'name = "semsearch"' "$1/pyproject.toml" 2>/dev/null
}

printf 'semsearch installer: local semantic code search for Claude Code\n'
note 'Needs about 5 GB of disk on Linux (PyTorch about 3.5 GB, the model about 1.5 GB).'
note 'An NVIDIA GPU with driver 580 or newer makes indexing much faster (optional).'

# --- 1. uv -----------------------------------------------------------------------------
step 'Checking uv'
if ! command -v uv >/dev/null 2>&1; then
    note 'uv is not installed. It comes from its official installer at https://astral.sh/uv'
    confirm 'Download and install uv now?' \
        || { note 'Stopped. Install uv yourself (https://docs.astral.sh/uv/) and run this again.'; exit 1; }
    command -v curl >/dev/null 2>&1 || fail 'curl is needed to download uv'
    # PATH in the shell profile is changed below, and only after asking
    curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh </dev/null \
        || fail 'the uv installer failed'
    PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 \
        || fail 'uv was installed but cannot be found; open a new terminal and run this again'
fi
note "$(uv --version </dev/null)"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
trap 'exit 130' INT TERM      # sh continues after a trap handler returns: stop instead

# --- 2. semsearch ----------------------------------------------------------------------
step 'Installing semsearch'
note 'The first install downloads PyTorch (about 3.5 GB on Linux); this takes a while.'
confirm 'Install semsearch now?' || { note 'Stopped. Nothing was installed.'; exit 1; }
src=${SEMSEARCH_INSTALL_SOURCE:-}
# from a clone only when this script is a real file there: under `curl | sh`, $0 is "sh" and
# the current directory is the user's, not ours
if [ -z "$src" ] && [ -f "$0" ] && is_source "$(dirname -- "$0")/.."; then
    src=$(cd -- "$(dirname -- "$0")/.." && pwd)
fi
if [ -n "$src" ]; then
    note "from $src"
else
    url="https://github.com/$REPO/archive/$REF.tar.gz"
    note "from $url"
    curl -fsSL "$url" -o "$tmp/src.tar.gz" </dev/null || fail "cannot download $url"
    mkdir "$tmp/src"
    tar -xzf "$tmp/src.tar.gz" -C "$tmp/src" || fail "cannot unpack $url"
    src=$(find "$tmp/src" -mindepth 1 -maxdepth 1 -type d | head -n 1)
fi
is_source "$src" || fail "$src is not a semsearch source folder"

uv export --project "$src" --frozen --no-dev --no-hashes --no-emit-project --quiet \
    -o "$tmp/constraints.txt" </dev/null || fail 'uv export failed'
uv tool install "$src" --python 3.12 --reinstall-package semsearch \
    --constraints "$tmp/constraints.txt" </dev/null || fail 'uv tool install failed'
bin=$(uv tool dir --bin </dev/null)
exe="$bin/semsearch"
[ -x "$exe" ] || fail "semsearch was installed but $exe is missing"
case ":$PATH:" in
    *":$bin:"*) ;;
    *)  note "$bin is not on your PATH, so new terminals will not find \`semsearch\`."
        if confirm 'Add it to your shell profile (uv tool update-shell)?'; then
            uv tool update-shell </dev/null
        fi ;;
esac

# --- 3. doctor -------------------------------------------------------------------------
step 'Checking the environment (semsearch doctor)'
"$exe" doctor --offline </dev/null || true

# --- 4. model --------------------------------------------------------------------------
step 'Embedding model'
if [ "$YES" = 1 ]; then set -- --yes; else set --; fi
"$exe" download-model "$@" <"$TTY" \
    || note 'The model is not downloaded; the first search downloads it, or run `semsearch download-model` later.'

# --- 5. Claude Code --------------------------------------------------------------------
step 'Registering with Claude Code'
if "$exe" install-claude "$@" <"$TTY"; then
    printf '\nDone. Open a new Claude Code session and ask about your code as usual.\n'
else
    printf '\nsemsearch is installed but not registered with Claude Code yet.\n'
    note 'Run `semsearch install-claude` when you are ready.'
fi
note 'Check the setup any time with: semsearch doctor'
