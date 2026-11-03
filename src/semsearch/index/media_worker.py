"""Media decoding in a separate, restricted process.

PyAV (FFmpeg), pdfium and Pillow parse files from the repository; a crafted file could crash,
hang or exploit one of them. They therefore run in a child process, `python -I -m
semsearch.index.media_worker`, that the daemon talks to over its stdin/stdout:

- the child limits its own memory (Windows job object, POSIX RLIMIT_AS) and on Windows may not
  start further processes; it dies instead of showing a crash dialog;
- every call has a time limit; a crash, hang or malformed answer kills the child and only the
  file being decoded is affected (the next call starts a fresh child);
- the child sends back JSON plus plain numpy arrays (`allow_pickle=False`), never pickles, and
  the daemon checks every value against the limits it asked for before using it.

It is not a full sandbox: the child runs as the same user and can read what the user can.

Framing: each message is an 8-byte big-endian length followed by the payload. Requests are
JSON. Responses are a 4-byte header length, a JSON header and the arrays it names, in order.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import queue
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any

import numpy as np

from semsearch.config import Settings
from semsearch.index import media
from semsearch.index.media import (
    AUDIO_HOP_S,
    AUDIO_RATE,
    AUDIO_WINDOW_S,
    IMAGE_MAX_SIDE,
    MIN_TAIL_S,
    PDF_RENDER_SIDE,
    PDF_TEXT_CHARS,
    VIDEO_FRAME_SIDE,
    VIDEO_SEGMENT_S,
    Limits,
    MediaChunk,
)

log = logging.getLogger(__name__)

MEMORY_LIMIT_BYTES = 2 << 30      # per child; decoding one chunk needs a few hundred MB at most
SEGMENT_TIMEOUT_S = 120.0         # reading page texts / durations of one file
DECODE_TIMEOUT_S = 120.0          # decoding one chunk (includes the child's start-up)
MAX_RESPONSE_BYTES = 256 << 20    # largest answer: a video segment is ~40 MB
MAX_HEADER_BYTES = 8 << 20        # JSON part: a 500-page PDF's texts are ~2 MB
LOG_NAME = "media-worker.log"
LOG_MAX_BYTES = 5_000_000
MAX_TEXT_CHARS = 1000             # symbol / preview strings from the child
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_CHUNK_KEYS = {"kind", "symbol", "preview", "text", "page", "t_start", "t_end"}


class MediaWorkerError(RuntimeError):
    """The worker crashed, hung or answered nonsense; it has been stopped."""


# --- framing ---------------------------------------------------------------------------------
def _read_exact(stream: IO[bytes], n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        block = stream.read(n - len(data))
        if not block:
            raise EOFError
        data.extend(block)
    return bytes(data)


def read_frame(stream: IO[bytes], limit: int) -> bytes:
    size = int.from_bytes(_read_exact(stream, 8), "big")
    if size > limit:
        raise ValueError(f"message of {size} bytes exceeds {limit}")
    return _read_exact(stream, size)


def write_frame(stream: IO[bytes], payload: bytes) -> None:
    stream.write(len(payload).to_bytes(8, "big") + payload)
    stream.flush()


def pack(header: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    head = json.dumps({**header, "arrays": list(arrays)}).encode("utf-8")
    buf = io.BytesIO()
    for arr in arrays.values():
        np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return len(head).to_bytes(4, "big") + head + buf.getvalue()


def unpack(payload: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Untrusted bytes -> (header, arrays). Raises ValueError on anything malformed."""
    if len(payload) < 4:
        raise ValueError("truncated response")
    size = int.from_bytes(payload[:4], "big")
    if size > MAX_HEADER_BYTES or 4 + size > len(payload):
        raise ValueError("bad header length")
    header = json.loads(payload[4:4 + size].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not an object")
    names = header.get("arrays", [])
    if (not isinstance(names, list) or len(names) > 2
            or not all(n in ("image", "audio", "frames") for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("bad array list")
    body = io.BytesIO(payload[4 + size:])
    arrays = {n: np.lib.format.read_array(body, allow_pickle=False) for n in names}
    if body.tell() != len(payload) - 4 - size:
        raise ValueError("trailing bytes in response")
    return header, arrays


# --- checks on what the child sends back ----------------------------------------------------
def _max_chunks(kind: str, limits: Limits) -> int:
    if kind == "image":
        return 1
    if kind == "pdf":
        return limits.max_pdf_pages
    hop = AUDIO_HOP_S if kind == "audio" else VIDEO_SEGMENT_S
    return math.ceil(limits.max_media_seconds / hop) + 1


def _str(value: Any, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("bad string field")
    return value


def _time(value: Any, limits: Limits) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("bad time field")
    t = float(value)
    if not math.isfinite(t) or t < 0 or t > limits.max_media_seconds + 1:
        raise ValueError("time out of range")
    return t


def check_chunks(raw: Any, kind: str, limits: Limits) -> list[MediaChunk]:
    if not isinstance(raw, list) or len(raw) > _max_chunks(kind, limits):
        raise ValueError("bad chunk list")
    out = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != _CHUNK_KEYS or item["kind"] != kind:
            raise ValueError("bad chunk")
        page = item["page"]
        if page is not None and (isinstance(page, bool) or not isinstance(page, int)
                                 or not 1 <= page <= limits.max_pdf_pages):
            raise ValueError("bad page")
        t_start, t_end = _time(item["t_start"], limits), _time(item["t_end"], limits)
        if (t_start is None) != (t_end is None) or (t_start is not None and t_end is not None
                                                    and t_start >= t_end):
            raise ValueError("bad time range")
        out.append(MediaChunk(kind, _str(item["symbol"], MAX_TEXT_CHARS),
                              _str(item["preview"], MAX_TEXT_CHARS),
                              _str(item["text"], PDF_TEXT_CHARS), page, t_start, t_end))
    return out


def _check_array(arr: np.ndarray, dtype: Any, ndim: int) -> None:
    if arr.dtype != dtype or arr.ndim != ndim or arr.size == 0:
        raise ValueError(f"expected a non-empty {ndim}-d {np.dtype(dtype).name} array")


def _check_rgb(arr: np.ndarray, side: int, ndim: int = 3) -> None:
    _check_array(arr, np.uint8, ndim)
    if arr.shape[-1] != 3 or max(arr.shape[-3], arr.shape[-2]) > side:
        raise ValueError(f"bad image shape {arr.shape}")


def _check_audio(arr: np.ndarray, seconds: float) -> None:
    _check_array(arr, np.float32, 1)
    if len(arr) > (seconds + 1) * AUDIO_RATE or not np.isfinite(arr).all():
        raise ValueError("bad audio samples")


def check_decoded(kind: str, arrays: dict[str, np.ndarray],
                  with_audio: bool) -> dict[str, np.ndarray]:
    names = set(arrays)
    if kind in ("image", "pdf"):
        if names != {"image"}:
            raise ValueError("expected one image")
        _check_rgb(arrays["image"], IMAGE_MAX_SIDE if kind == "image" else PDF_RENDER_SIDE)
    elif kind == "audio":
        if names != {"audio"}:
            raise ValueError("expected audio samples")
        _check_audio(arrays["audio"], AUDIO_WINDOW_S + MIN_TAIL_S)
    elif kind == "video":
        if "frames" not in names or not names <= {"frames", "audio"}:
            raise ValueError("expected video frames")
        frames = arrays["frames"]
        _check_rgb(frames, VIDEO_FRAME_SIDE, ndim=4)
        if len(frames) > VIDEO_SEGMENT_S + MIN_TAIL_S + 1:
            raise ValueError("too many frames")
        if "audio" in arrays:
            if not with_audio:
                raise ValueError("audio was not asked for")
            _check_audio(arrays["audio"], VIDEO_SEGMENT_S + MIN_TAIL_S)
    else:
        raise ValueError(f"unknown media kind {kind!r}")
    return arrays


# --- the daemon side -------------------------------------------------------------------------
class MediaDecoder:
    """Runs media.segment() / media.decode() in a child process. Not thread-safe: one per
    indexing thread. The child starts on first use and stops with close()."""

    def __init__(self, settings: Settings, command: list[str] | None = None) -> None:
        self.settings = settings
        self.limits = Limits(settings.max_pdf_pages, settings.max_media_seconds)
        self.command = command or [sys.executable, "-I", "-m", "semsearch.index.media_worker"]
        self._proc: subprocess.Popen[bytes] | None = None
        self._answers: queue.Queue[bytes | None] = queue.Queue()

    # public API
    def segment(self, path: Path, kind: str) -> list[MediaChunk]:
        header, _ = self._call({"op": "segment", "path": str(path), "kind": kind},
                               SEGMENT_TIMEOUT_S)
        return self._guard(lambda: check_chunks(header.get("chunks"), kind, self.limits))

    def decode(self, path: Path, chunk: MediaChunk, with_audio: bool) -> dict[str, np.ndarray]:
        _, arrays = self._call({"op": "decode", "path": str(path), "chunk": asdict(chunk),
                                "with_audio": with_audio}, DECODE_TIMEOUT_S)
        return self._guard(lambda: check_decoded(chunk.kind, arrays, with_audio))

    def load(self, path: Path, chunk: MediaChunk, with_audio: bool) -> dict[str, Any]:
        """Like media.load(), decoded in the child."""
        return media.assemble(path, chunk, self.decode(path, chunk, with_audio))

    def release(self) -> None:
        """Let the child close the PDF it keeps open (Windows locks open files)."""
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(MediaWorkerError):
                self._call({"op": "release"}, DECODE_TIMEOUT_S)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(OSError, ValueError):
            if proc.stdin is not None:
                proc.stdin.close()           # EOF: the child exits
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def __enter__(self) -> MediaDecoder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # internals
    def _guard(self, check: Any) -> Any:
        try:
            return check()
        except ValueError as exc:            # the child is misbehaving: do not trust it again
            self._kill()
            raise MediaWorkerError(f"invalid answer from media worker: {exc}") from exc

    def _start(self) -> subprocess.Popen[bytes]:
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.settings.log_dir / LOG_NAME
        with contextlib.suppress(OSError):
            if log_path.stat().st_size > LOG_MAX_BYTES:
                log_path.unlink()
        with open(log_path, "ab") as err:
            proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=err, cwd=str(self.settings.log_dir), close_fds=True,
                                    creationflags=NO_WINDOW)
        self._answers = queue.Queue()
        threading.Thread(target=self._read_answers, args=(proc, self._answers),
                         name="media-worker-reader", daemon=True).start()
        assert proc.stdin is not None
        write_frame(proc.stdin, json.dumps({
            "max_pdf_pages": self.limits.max_pdf_pages,
            "max_media_seconds": self.limits.max_media_seconds,
            "memory_bytes": MEMORY_LIMIT_BYTES}).encode("utf-8"))
        return proc

    @staticmethod
    def _read_answers(proc: subprocess.Popen[bytes], answers: queue.Queue[bytes | None]) -> None:
        assert proc.stdout is not None
        try:
            while True:
                answers.put(read_frame(proc.stdout, MAX_RESPONSE_BYTES))
        except (EOFError, OSError, ValueError) as exc:
            if isinstance(exc, ValueError):          # oversized answer: stop the child
                with contextlib.suppress(OSError):
                    proc.kill()
            answers.put(None)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            with contextlib.suppress(OSError):
                proc.kill()
            proc.wait()

    def _call(self, request: dict[str, Any], timeout: float
              ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = self._start()
        proc = self._proc
        try:
            assert proc.stdin is not None
            write_frame(proc.stdin, json.dumps(request).encode("utf-8"))
            answer = self._answers.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            raise MediaWorkerError(f"media worker timed out after {timeout:.0f}s") from None
        except OSError:                          # broken pipe: it died while we wrote
            answer = None
        if answer is None:
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
            self._proc = None
            raise MediaWorkerError(f"media worker exited with code {code}")
        try:
            header, arrays = unpack(answer)
        except Exception as exc:     # malformed JSON / npy data (also RecursionError)
            self._kill()
            raise MediaWorkerError(f"invalid answer from media worker: {exc}") from exc
        if not header.get("ok"):
            # an ordinary decoding error (corrupt file): the child itself is fine
            raise ValueError(str(header.get("error", "media worker error"))[:500])
        return header, arrays


# --- the child side --------------------------------------------------------------------------
def _limit_memory(limit: int) -> None:
    if sys.platform == "win32":
        _windows_job(limit)
        return
    try:
        import resource
        # address space, not resident memory: leave room for library mappings and thread stacks
        resource.setrlimit(resource.RLIMIT_AS, (limit * 2, limit * 2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError) as exc:
        log.warning("cannot limit media worker memory: %s", exc)


def _windows_job(limit: int) -> None:
    """Put this process in a job object: memory capped, no child processes, and an unhandled
    exception kills it instead of opening a crash dialog that would hang it."""
    assert sys.platform == "win32"   # also tells type checkers on other systems to skip the rest
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job_limit_active_process = 0x8
    job_limit_process_memory = 0x100
    job_limit_die_on_unhandled_exception = 0x400
    job_extended_limit_information = 9
    sem_failcriticalerrors, sem_nogpfaulterrorbox = 0x1, 0x2

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetErrorMode(sem_failcriticalerrors | sem_nogpfaulterrorbox)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                 ctypes.c_void_p, wintypes.DWORD]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = (job_limit_active_process | job_limit_process_memory
                                             | job_limit_die_on_unhandled_exception)
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = limit
    if not (job and kernel32.SetInformationJobObject(job, job_extended_limit_information,
                                                     ctypes.byref(info), ctypes.sizeof(info))
            and kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())):
        log.warning("cannot restrict the media worker (error %s)", ctypes.get_last_error())
    # the handle stays open for the life of the process: closing it would end the limits


def _handle(request: dict[str, Any], limits: Limits) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    op = request.get("op")
    if op == "segment":
        chunks = media.segment(Path(request["path"]), request["kind"], limits)
        return {"ok": True, "chunks": [asdict(c) for c in chunks]}, {}
    if op == "decode":
        chunk = MediaChunk(**request["chunk"])
        return {"ok": True}, media.decode(Path(request["path"]), chunk,
                                          bool(request["with_audio"]))
    if op == "release":
        media.release()
        return {"ok": True}, {}
    raise ValueError(f"unknown op {op!r}")


def serve(stdin: IO[bytes], stdout: IO[bytes]) -> None:
    init = json.loads(read_frame(stdin, MAX_HEADER_BYTES))
    _limit_memory(int(init["memory_bytes"]))
    limits = Limits(int(init["max_pdf_pages"]), float(init["max_media_seconds"]))
    while True:
        try:
            request = json.loads(read_frame(stdin, MAX_HEADER_BYTES))
        except EOFError:
            return
        try:
            header, arrays = _handle(request, limits)
        except Exception as exc:     # MemoryError included: report it, keep serving
            header, arrays = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}, {}
        write_frame(stdout, pack(header, arrays))


def main() -> None:
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr      # a stray print() must never corrupt the protocol stream
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s media-worker %(levelname)s %(message)s")
    try:
        serve(stdin, stdout)
    finally:
        media.release()


if __name__ == "__main__":
    main()
