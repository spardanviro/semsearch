"""After a fresh install: index a tiny project with the real model, then check a search finds the
right file. Runs the installed `semsearch` command, not the source tree."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

INDEX_TIMEOUT_S = 1800   # CPU-only runners: model load plus first index
POLL_S = 5

FILES = {
    "net/fetch.py": '''
import time

def download_with_retry(url, attempts=3):
    """Download a URL; on a network error wait and try again, up to `attempts` times."""
    for attempt in range(attempts):
        try:
            return http_get(url)
        except ConnectionError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}")
''',
    "game/inventory.py": '''
import json

def save_inventory(player, path):
    """Write the player's items and gold to a JSON save file."""
    with open(path, "w") as f:
        json.dump({"items": player.items, "gold": player.gold}, f)
''',
    "util/geometry.py": '''
import math

def circle_area(radius):
    return math.pi * radius * radius
''',
}
CASES = {
    "where are failed downloads retried": "net/fetch.py",
    "code that saves the player inventory to disk": "game/inventory.py",
}


def semsearch(exe: str, *args: str) -> dict:
    out = subprocess.run([exe, *args, "--json"], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def wait_indexed(exe: str, root: str) -> dict:
    deadline = time.monotonic() + INDEX_TIMEOUT_S
    status: dict = {}
    while time.monotonic() < deadline:
        status = semsearch(exe, "status", "--root", root)
        if status.get("last_error"):
            sys.exit(f"indexing failed: {status['last_error']}")
        if not status.get("indexing") and status.get("chunks"):
            return status
        time.sleep(POLL_S)
    sys.exit(f"not indexed after {INDEX_TIMEOUT_S}s: {status}")


def main() -> int:
    exe = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "demo"
        for rel, text in FILES.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text.lstrip(), encoding="utf-8")
        try:
            status = wait_indexed(exe, str(root))
            print(f"indexed {status['files']} files, {status['chunks']} chunks with {status['model']}")
            failed = 0
            for query, want in CASES.items():
                results = semsearch(exe, "search", "--root", str(root), query)["results"]
                top = results[0]["path"] if results else None
                print(f"{'ok  ' if top == want else 'FAIL'} {query!r} -> {top}")
                failed += top != want
            return 1 if failed else 0
        finally:
            subprocess.run([exe, "daemon", "stop"], check=False)   # releases the index files


if __name__ == "__main__":
    sys.exit(main())
