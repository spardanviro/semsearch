"""Hardening from docs/AUDIT.md "follow-up": media decoding in a restricted child process,
nested .gitignore files without git/rg, and daemon request limits."""
from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from semsearch import client as client_mod
from semsearch.client import DaemonClient
from semsearch.config import Settings
from semsearch.daemon.app import DaemonState, create_app
from semsearch.daemon.limits import MAX_BODY_BYTES, RateLimiter
from semsearch.embed.base import FakeEmbedder
from semsearch.index import indexer as indexer_mod
from semsearch.index import media, media_worker, scanner
from semsearch.index.media import MediaChunk
from semsearch.index.media_worker import MediaDecoder, MediaWorkerError, pack, unpack
from semsearch.index.project import Project


def _settings(tmp_path: Path, **kw: object) -> Settings:
    return replace(Settings(home=tmp_path / "home", embedder="fake"), **kw)  # type: ignore[arg-type]


def _png(path: Path, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (10, 200, 30)).save(path)
    return path


def _fake_worker(tmp_path: Path, body: str) -> list[str]:
    """A stand-in child speaking the real protocol; `body` handles one request `req`."""
    script = tmp_path / "fake_worker.py"
    script.write_text(textwrap.dedent("""
        import json, os, sys, time
        import numpy as np
        from semsearch.index.media_worker import pack, read_frame, write_frame, serve
        stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
        read_frame(stdin, 1 << 20)                      # init
        while True:
            try:
                req = json.loads(read_frame(stdin, 1 << 20))
            except EOFError:
                break
    """) + textwrap.indent(textwrap.dedent(body), " " * 4), encoding="utf-8")
    return [sys.executable, str(script)]


CHUNK = MediaChunk("image", "a.png", "[image 64x64]")


# --- media worker: the real child ------------------------------------------------------------
def test_worker_decodes_the_same_as_in_process(tmp_path: Path) -> None:
    png = _png(tmp_path / "a.png")
    with MediaDecoder(_settings(tmp_path)) as dec:
        chunks = dec.segment(png, "image")
        raw = dec.decode(png, chunks[0], with_audio=False)

    assert chunks == media.segment(png, "image", Settings())
    np.testing.assert_array_equal(raw["image"], media.decode(png, chunks[0], False)["image"])


def test_worker_reports_a_corrupt_file_and_keeps_running(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png at all")
    png = _png(tmp_path / "a.png")
    with MediaDecoder(_settings(tmp_path)) as dec:
        with pytest.raises(ValueError, match="UnidentifiedImageError"):
            dec.segment(bad, "image")
        proc = dec._proc
        assert dec.segment(png, "image")         # same child still serves

    assert proc is not None and proc.poll() is not None    # closed with the decoder


def test_worker_limits_its_memory_and_cannot_start_processes(tmp_path: Path) -> None:
    code = textwrap.dedent("""
        import subprocess, sys
        from semsearch.index.media_worker import _limit_memory
        _limit_memory(300 << 20)
        try:
            blob = bytearray(1 << 30)
            print("allocated")
        except MemoryError:
            print("memory-capped")
        if sys.platform == "win32":
            try:
                subprocess.run([sys.executable, "-c", "0"], check=True)
                print("spawned")
            except OSError:
                print("spawn-blocked")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=60).stdout

    assert "memory-capped" in out
    if sys.platform == "win32":
        assert "spawn-blocked" in out


# --- media worker: misbehaving children ------------------------------------------------------
def test_worker_crash_is_reported_and_the_next_call_restarts_it(tmp_path: Path) -> None:
    dec = MediaDecoder(_settings(tmp_path), command=_fake_worker(tmp_path, "os._exit(7)"))

    with pytest.raises(MediaWorkerError, match="code 7"):
        dec.decode(tmp_path / "a.png", CHUNK, with_audio=False)
    with pytest.raises(MediaWorkerError, match="code 7"):    # a fresh child, same fate
        dec.decode(tmp_path / "a.png", CHUNK, with_audio=False)
    dec.close()


def test_worker_hang_is_killed_after_the_timeout(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_worker, "DECODE_TIMEOUT_S", 1.0)
    dec = MediaDecoder(_settings(tmp_path), command=_fake_worker(tmp_path, "time.sleep(60)"))

    with pytest.raises(MediaWorkerError, match="timed out"):
        dec.decode(tmp_path / "a.png", CHUNK, with_audio=False)

    assert dec._proc is None


@pytest.mark.parametrize("answer", [
    "write_frame(stdout, b'garbage')",
    # an object array would need pickle to load: refused, never unpickled
    "write_frame(stdout, pack({'ok': True}, {'image': np.array([object()], dtype=object)}))",
    "write_frame(stdout, pack({'ok': True}, {'image': np.zeros((5000, 10, 3), np.uint8)}))",
    "write_frame(stdout, pack({'ok': True}, {'image': np.zeros((8, 8, 3), np.float64)}))",
    "write_frame(stdout, pack({'ok': True}, {'audio': np.zeros(10, np.float32)}))",
    "stdout.write((1 << 40).to_bytes(8, 'big')); stdout.flush(); time.sleep(30)",
])
def test_worker_malformed_or_out_of_range_answers_are_rejected(tmp_path: Path,
                                                               answer: str) -> None:
    dec = MediaDecoder(_settings(tmp_path), command=_fake_worker(tmp_path, answer))

    with pytest.raises(MediaWorkerError):
        dec.decode(tmp_path / "a.png", CHUNK, with_audio=False)

    assert dec._proc is None          # a misbehaving child is never reused
    dec.close()


def test_worker_segment_answers_are_checked(tmp_path: Path) -> None:
    bad = {"kind": "pdf", "symbol": "p.1", "preview": "x", "text": "", "page": 10_000,
           "t_start": None, "t_end": None}
    body = f"write_frame(stdout, pack({{'ok': True, 'chunks': [{bad!r}]}}, {{}}))"
    dec = MediaDecoder(_settings(tmp_path, max_pdf_pages=5),
                       command=_fake_worker(tmp_path, body))

    with pytest.raises(MediaWorkerError, match="bad page"):
        dec.segment(tmp_path / "a.pdf", "pdf")
    dec.close()


def test_pack_unpack_round_trip() -> None:
    frames = np.zeros((3, 4, 5, 3), np.uint8)
    header, arrays = unpack(pack({"ok": True}, {"frames": frames}))

    assert header == {"ok": True, "arrays": ["frames"]}
    np.testing.assert_array_equal(arrays["frames"], frames)
    with pytest.raises(ValueError):
        unpack(pack({"ok": True}, {"frames": frames}) + b"x")


def test_indexer_skips_the_rest_of_a_file_that_crashes_the_worker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    _png(root / "good.png")
    _png(root / "poison.png", size=80)   # different bytes: no shared chunk hash
    body = """
        if req.get("op") == "decode" and "poison" in req["path"]:
            os._exit(9)
        from semsearch.index.media_worker import _handle
        from semsearch.index.media import Limits
        header, arrays = _handle(req, Limits(500, 3600.0))
        write_frame(stdout, pack(header, arrays))
    """
    command = _fake_worker(tmp_path, body)
    monkeypatch.setattr(indexer_mod, "MediaDecoder",
                        lambda settings: MediaDecoder(settings, command=command))
    settings = replace(_settings(tmp_path), media=frozenset({"image"}), scan_cooldown_s=0.0)

    p = Project(root, settings, FakeEmbedder())
    try:
        assert p.wait_idle(60)
        with p.db_lock:
            files = p.store.files()
        assert set(files) == {"good.png", "poison.png"}
        assert p.status()["chunks"] == 1        # good.png only; the index still updated
    finally:
        p.close()


# --- nested .gitignore without git or rg -----------------------------------------------------
LAYOUT = {
    ".gitignore": "*.log\n!keep.log\nbuild/\n/rootonly.txt\n",
    "a.py": "", "x.log": "", "keep.log": "", "rootonly.txt": "", "sub/rootonly.txt": "",
    "build/out.py": "", "sub/.gitignore": "!x.log\nsecret/\n*.tmp\n", "sub/x.log": "",
    "sub/y.tmp": "", "sub/secret/s.py": "", "sub/deep/.gitignore": "!z.tmp\n",
    "sub/deep/z.tmp": "", "sub/deep/w.py": "", "other/y.tmp": "", "other/build/q.py": "",
}
EXPECTED = {".gitignore", "a.py", "keep.log", "sub/rootonly.txt", "sub/.gitignore",
            "sub/x.log", "sub/deep/.gitignore", "sub/deep/z.tmp", "sub/deep/w.py",
            "other/y.tmp"}


@pytest.fixture
def layout(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel, text in LAYOUT.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def test_walk_honours_nested_gitignore_files(layout: Path) -> None:
    assert set(scanner._list_with_walk(layout)) == EXPECTED


def test_walk_agrees_with_git(layout: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not installed")
    subprocess.run([git, "init", "-q"], cwd=layout, check=True)
    listed = subprocess.run([git, "ls-files", "-co", "--exclude-standard"], cwd=layout,
                            capture_output=True, text=True, check=True).stdout.split()

    assert set(scanner._list_with_walk(layout)) == set(listed)


def test_walk_reads_git_info_exclude_below_gitignore(layout: Path) -> None:
    (layout / ".git" / "info").mkdir(parents=True)
    (layout / ".git" / "info" / "exclude").write_text("a.py\nkeep.log\n", encoding="utf-8")

    listed = set(scanner._list_with_walk(layout))

    assert "a.py" not in listed
    assert "keep.log" in listed       # .gitignore's !keep.log comes later and wins


def test_hook_check_without_git_matches_the_walk(layout: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scanner.shutil, "which", lambda name: None)

    wrong = [rel for rel in LAYOUT if scanner.is_gitignored(layout, rel) == (rel in EXPECTED)]

    assert wrong == []


# --- daemon request limits -------------------------------------------------------------------
@pytest.fixture
def state(tmp_path: Path) -> Iterator[DaemonState]:
    st = DaemonState(settings=_settings(tmp_path), embedder=FakeEmbedder())
    yield st
    st.close()


def _client(state: DaemonState, auth: bool = True) -> TestClient:
    c = TestClient(create_app(state), base_url="http://127.0.0.1")
    if auth:
        c.headers["Authorization"] = f"Bearer {state.token}"
    return c


def test_oversized_body_is_rejected(state: DaemonState, tmp_path: Path) -> None:
    body = b'{"root": "' + b"a" * (MAX_BODY_BYTES + 10) + b'"}'

    r = _client(state).post("/status", content=body,
                            headers={"content-type": "application/json"})

    assert r.status_code == 413


def test_oversized_chunked_body_is_rejected(state: DaemonState) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"root": "'
        for _ in range(20):
            yield b"a" * (MAX_BODY_BYTES // 10)
        yield b'"}'

    r = _client(state).post("/status", content=chunks(),
                            headers={"content-type": "application/json"})

    assert r.status_code == 413


def test_rate_limit_answers_429_with_retry_after(state: DaemonState, tmp_path: Path) -> None:
    now = [0.0]
    state.limiter = RateLimiter(rate=1.0, burst=2, clock=lambda: now[0])
    c = _client(state)
    root = tmp_path / "proj"
    root.mkdir()

    codes = [c.post("/status", json={"root": str(root)}).status_code for _ in range(3)]
    limited = c.post("/status", json={"root": str(root)})
    health = c.get("/health")
    now[0] += 1.0
    later = c.post("/status", json={"root": str(root)})

    assert codes == [200, 200, 429]
    assert limited.headers["retry-after"] == "1"
    assert health.status_code == 200          # liveness checks are never throttled
    assert later.status_code == 200


def test_requests_without_the_token_do_not_use_up_the_budget(state: DaemonState,
                                                             tmp_path: Path) -> None:
    state.limiter = RateLimiter(rate=0.001, burst=1)
    stranger = _client(state, auth=False)
    root = tmp_path / "proj"
    root.mkdir()

    assert all(stranger.post("/status", json={"root": str(root)}).status_code == 401
               for _ in range(5))
    assert _client(state).post("/status", json={"root": str(root)}).status_code == 200


def test_rate_limiter_refills_over_time() -> None:
    now = [0.0]
    limiter = RateLimiter(rate=2.0, burst=1, clock=lambda: now[0])

    assert limiter.wait() == 0
    assert limiter.wait() == pytest.approx(0.5)
    now[0] += 0.5
    assert limiter.wait() == 0
    with pytest.raises(ValueError):
        RateLimiter(rate=0, burst=1)


def test_client_waits_out_a_429_and_retries(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    answers = [httpx.Response(429, headers={"retry-after": "0"}, json={"detail": "slow"}),
               httpx.Response(200, json={"ok": True})]
    monkeypatch.setattr(client_mod, "_http", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    c = DaemonClient(_settings(tmp_path))
    monkeypatch.setattr(c, "ensure", lambda: {"port": 1, "token": "t"})

    assert c.request("/status", {"root": "x"}) == {"ok": True}
    assert answers == []


def test_client_gives_up_when_the_wait_exceeds_its_timeout(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod, "_http", lambda *a, **k: httpx.Response(
        429, headers={"retry-after": "3"}, json={"detail": "too many requests"}))
    c = DaemonClient(_settings(tmp_path))
    monkeypatch.setattr(c, "ensure", lambda: {"port": 1, "token": "t"})

    with pytest.raises(client_mod.DaemonUnavailable, match="429"):
        c.request("/notify", {"root": "x", "paths": []}, timeout=0.5)

