"""Media indexing (phase 2/3): scanning, segmentation, decoding, indexing, search, formatting.

All media are generated on the fly (PIL, a hand-written PDF, wave, PyAV), so the tests need no
fixtures and no model; the real-model check lives in test_st_backend (SEMSEARCH_GPU_TESTS=1).
"""
from __future__ import annotations

import math
import struct
import wave
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semsearch.config import Settings
from semsearch.embed.base import FakeEmbedder
from semsearch.formatting import format_search
from semsearch.index import media
from semsearch.index.media import MediaChunk, windows
from semsearch.index.project import Project
from semsearch.index.scanner import classify, scan


# --- generators ------------------------------------------------------------------------------
def make_png(path: Path, size: int = 96, color: tuple[int, int, int] = (200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)
    return path


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Minimal valid PDF with one Helvetica text line per page."""
    objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    kids = []
    for text in pages:
        stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
        content_id = len(objs) + 2
        page_id = len(objs) + 1
        kids.append(f"{page_id} 0 R")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents "
                    f"{content_id} 0 R /Resources << /Font << /F1 << /Type /Font /Subtype "
                    f"/Type1 /BaseFont /Helvetica >> >> >> >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


def make_wav(path: Path, seconds: float, freq: float = 440.0, rate: int = 22050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = (int(12000 * math.sin(2 * math.pi * freq * i / rate))
                  for i in range(int(seconds * rate)))
        w.writeframes(b"".join(struct.pack("<h", s) for s in frames))
    return path


def make_video(path: Path, seconds: int, fps: int = 2) -> Path:
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, fps)
        for i in range(seconds * fps):
            img = np.full((48, 64, 3), (i * 7) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


ALL = frozenset({"image", "pdf", "audio", "video"})


# --- pure helpers ----------------------------------------------------------------------------
@pytest.mark.parametrize(("duration", "expected"), [
    (0, []),
    (12, [(0, 12)]),
    (30, [(0, 30)]),
    (52, [(0, 30), (25, 52)]),
    (57, [(0, 30), (25, 55), (50, 57)]),      # a 7 s tail stays its own window
    (53.5, [(0, 30), (25, 53.5)]),
])
def test_media_windows_cover_duration_and_merge_short_tail(duration: float,
                                                            expected: list[tuple[float, float]]) -> None:
    assert windows(duration, 30, 25) == expected


def test_media_windows_tail_shorter_than_minimum_joins_previous() -> None:
    assert windows(42, 20, 20) == [(0, 20), (20, 42)]      # 2 s tail merged


def test_classify_indexes_media_only_when_enabled() -> None:
    assert classify("art/hero.png") is None
    assert classify("art/hero.png", frozenset({"image"})) == ("image", "png")
    assert classify("docs/a.PDF", ALL) == ("pdf", "pdf")
    assert classify("sfx/hit.ogg", frozenset({"image"})) is None
    assert classify("src/main.cs", ALL) == ("code", "csharp")


def test_settings_parse_media(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMSEARCH_MEDIA", "Image, pdf")
    assert Settings().media == {"image", "pdf"}
    monkeypatch.setenv("SEMSEARCH_MEDIA", "all")
    assert Settings().media == ALL
    monkeypatch.setenv("SEMSEARCH_MEDIA", "none")
    assert Settings().media == frozenset()


@pytest.mark.parametrize(("env", "message"), [
    ({"SEMSEARCH_MEDIA": "image,holograms"}, "unknown media kinds"),
    ({"SEMSEARCH_MEDIA": "image", "SEMSEARCH_MODEL": "BAAI/bge-m3"}, "EmbeddingGemma"),
    ({"SEMSEARCH_MAX_PDF_PAGES": "0"}, "must be > 0"),
])
def test_settings_reject_bad_media_options(monkeypatch: pytest.MonkeyPatch,
                                           env: dict[str, str], message: str) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    with pytest.raises(ValueError, match=message):
        Settings()


# --- segmentation and decoding ---------------------------------------------------------------
def test_segment_image_and_skip_icons(tmp_path: Path) -> None:
    big = make_png(tmp_path / "big.png", 96)
    icon = make_png(tmp_path / "icon.png", 16)

    [chunk] = media.segment(big, "image", Settings())
    item = media.load(big, chunk, with_audio=False)

    assert chunk.preview == "[image 96x96]" and chunk.symbol == "big.png"
    assert media.segment(icon, "image", Settings()) == []
    assert item["image"].size == (96, 96)


def test_segment_pdf_pages_with_text_and_limit(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "doc.pdf", ["treasure map legend", "second page notes", "third"])

    chunks = media.segment(pdf, "pdf", replace(Settings(), max_pdf_pages=2))
    item = media.load(pdf, chunks[0], with_audio=False)

    assert [c.page for c in chunks] == [1, 2] and chunks[0].symbol == "p.1"
    assert chunks[0].text == "treasure map legend"
    assert item["text"].startswith("title: doc.pdf p.1 | text: treasure map legend")
    assert item["text"].endswith("<|image|>") and max(item["image"][0].size) == media.PDF_RENDER_SIDE


def test_segment_and_load_audio_windows(tmp_path: Path) -> None:
    wav = make_wav(tmp_path / "loop.wav", 52)

    chunks = media.segment(wav, "audio", Settings())
    item = media.load(wav, chunks[1], with_audio=False)

    assert [(c.t_start, c.t_end) for c in chunks] == [(0, 30), (25, 52)]
    assert chunks[1].symbol == "00:25-00:52" and "of 00:52" in chunks[1].preview
    samples = item["audio"]["array"]
    assert item["audio"]["sampling_rate"] == 16000 and samples.dtype == np.float32
    assert abs(len(samples) - 27 * 16000) < 1600            # 27 s at 16 kHz, mono


def test_segment_and_load_video_frames_at_one_fps(tmp_path: Path) -> None:
    clip = make_video(tmp_path / "clip.mp4", 24)

    chunks = media.segment(clip, "video", Settings())
    item = media.load(clip, chunks[0], with_audio=True)    # no audio track -> frames only

    assert [(c.t_start, c.t_end) for c in chunks] == [(0, 24)]   # 4 s tail merged
    assert set(item) == {"video"}
    frames = item["video"]
    assert frames.ndim == 4 and frames.shape[3] == 3 and 23 <= frames.shape[0] <= 24


def make_offset_mkv(path: Path, seconds: int, offset_s: int, video: bool) -> Path:
    """mkv whose timestamps start at `offset_s` (common for recordings). Each second carries
    its index: audio as a constant level k/100, video as a gray level 5*k."""
    import av
    with av.open(str(path), "w", format="matroska") as out:
        if video:
            stream = out.add_stream("mpeg4", rate=2)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            stream.time_base = Fraction(1, 2)
            for i in range(seconds * 2):
                img = np.full((48, 64, 3), 5 * (i // 2), dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame.pts = offset_s * 2 + i
                out.mux(stream.encode(frame))
        else:
            stream = out.add_stream("pcm_s16le", rate=16000, layout="mono")
            stream.time_base = Fraction(1, 16000)
            for k in range(seconds):
                level = int(32767 * k / 100)
                frame = av.AudioFrame.from_ndarray(np.full((1, 16000), level, dtype=np.int16),
                                                   format="s16", layout="mono")
                frame.sample_rate, frame.pts = 16000, (offset_s + k) * 16000
                out.mux(stream.encode(frame))
        out.mux(stream.encode())
    return path


def test_audio_windows_respect_a_nonzero_start_time(tmp_path: Path) -> None:
    # Regression: windows counted from 0 but packets from 100 s -> every window got the start.
    mkv = make_offset_mkv(tmp_path / "rec.mkv", 60, offset_s=100, video=False)

    chunks = media.segment(mkv, "audio", Settings())
    second = media.load(mkv, chunks[1], with_audio=False)["audio"]["array"]

    assert chunks[1].t_start == 25
    assert second[8000] == pytest.approx(0.25, abs=0.01)      # 0.5 s into the window = second 25


def test_video_frames_respect_a_nonzero_start_time(tmp_path: Path) -> None:
    mkv = make_offset_mkv(tmp_path / "rec.mkv", 44, offset_s=100, video=True)

    chunks = media.segment(mkv, "video", Settings())
    frames = media.load(mkv, chunks[1], with_audio=False)["video"]

    assert chunks[1].t_start == 20
    assert abs(int(frames[0].mean()) - 5 * 20) <= 4               # first frame is second 20


def test_segment_image_skips_decompression_bombs(tmp_path: Path) -> None:
    huge = tmp_path / "huge.png"
    Image.new("1", (9000, 8000)).save(huge)                      # 72 Mpx, tiny on disk

    assert media.segment(huge, "image", Settings()) == []


def test_max_media_seconds_limits_indexed_duration(tmp_path: Path) -> None:
    wav = make_wav(tmp_path / "long.wav", 70)

    chunks = media.segment(wav, "audio", replace(Settings(), max_media_seconds=40))

    assert chunks[-1].t_end == 40


def test_chunk_locator_distinguishes_positions() -> None:
    a = MediaChunk("audio", "00:00-00:30", "", t_start=0, t_end=30)
    b = MediaChunk("audio", "00:25-00:52", "", t_start=25, t_end=52)

    assert a.locator != b.locator


# --- indexing + search (fake embedder) -------------------------------------------------------
@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("def main():\n    return 'hello world'\n", "utf-8")
    make_png(root / "art" / "hero.png", 96, (10, 200, 10))
    make_pdf(root / "docs" / "manual.pdf", ["install the treasure map plugin", "credits"])
    make_wav(root / "sfx" / "loop.wav", 12)
    make_wav(root / "sfx" / "other.wav", 12, freq=880)     # same length, different sound
    return root


@pytest.fixture
def media_settings(tmp_path: Path) -> Settings:
    return replace(Settings(home=tmp_path / "home", embedder="fake"), scan_cooldown_s=0.0,
                   search_wait_s=10.0, media=ALL)


@pytest.fixture
def media_project(media_root: Path, media_settings: Settings) -> Iterator[Project]:
    p = Project(media_root, media_settings, FakeEmbedder())
    assert p.wait_idle(30)
    yield p
    p.close()


def test_scan_skips_media_when_disabled(media_root: Path) -> None:
    files = scan(media_root, Settings())

    assert set(files) == {"src/main.py"}


def test_project_indexes_media_with_pages_and_times(media_project: Project) -> None:
    assert media_project.status()["files"] == 5
    results, _ = media_project.search("treasure map plugin install", k=3)
    pdf = next(r for r in results if r.modality == "pdf")
    audio, _ = media_project.search("loop", k=5, modality="audio")

    assert pdf.page == 1 and pdf.symbol == "p.1" and pdf.path == "docs/manual.pdf"
    assert {r.path for r in audio} == {"sfx/loop.wav", "sfx/other.wav"}   # not merged as copies
    assert all(r.t_start == 0 and r.t_end == 12 for r in audio)


def test_project_media_unchanged_is_not_reembedded(media_root: Path,
                                                   media_settings: Settings) -> None:
    first = FakeEmbedder()
    p = Project(media_root, media_settings, first)
    assert p.wait_idle(30)
    p.close()

    second = FakeEmbedder()
    p = Project(media_root, media_settings, second)
    try:
        assert p.wait_idle(30)
        assert first.media_encoded == 5 and second.media_encoded == 0
    finally:
        p.close()


def test_project_identical_media_copies_collapse(media_root: Path, media_settings: Settings) -> None:
    (media_root / "art" / "copy").mkdir()
    (media_root / "art" / "copy" / "hero.png").write_bytes((media_root / "art" / "hero.png").read_bytes())
    embedder = FakeEmbedder()
    p = Project(media_root, media_settings, embedder)
    try:
        assert p.wait_idle(30)
        results, _ = p.search("hero", k=10, modality="image")

        assert len(results) == 1 and results[0].duplicates
        assert embedder.media_encoded == 5        # the copy reused the original's vector
    finally:
        p.close()


def test_project_corrupt_media_is_recorded_and_others_survive(media_root: Path,
                                                              media_settings: Settings) -> None:
    (media_root / "art" / "broken.png").write_bytes(b"not an image at all")
    p = Project(media_root, media_settings, FakeEmbedder())
    try:
        assert p.wait_idle(30)
        assert p.status()["last_error"] is None
        assert "art/broken.png" in p.store.files()            # recorded: not retried every scan
        results, _ = p.search("hero", k=10, modality="image")
        assert [r.path for r in results] == ["art/hero.png"]
    finally:
        p.close()


def test_project_model_error_on_one_media_input_skips_only_that_input(
        media_root: Path, media_settings: Settings) -> None:
    # Regression: one failing media batch aborted the whole update (nothing got indexed).
    class Picky(FakeEmbedder):
        def encode_media(self, items: list[dict[str, object]]) -> np.ndarray:
            if any("audio" in item for item in items):
                raise ValueError("model rejected the input")
            return super().encode_media(items)

    p = Project(media_root, media_settings, Picky())
    try:
        assert p.wait_idle(30)
        assert p.status()["last_error"] is None
        modalities = {r.modality for r in p.search("anything", k=20)[0]}
        assert {"code", "image", "pdf"} <= modalities and "audio" not in modalities
    finally:
        p.close()


def test_project_transient_media_failure_is_retried_next_scan(
        media_root: Path, media_settings: Settings) -> None:
    # Regression: chunks dropped by a GPU out-of-memory error were never retried.
    class OutOfMemoryError(RuntimeError):
        pass

    class FlakyGpu(FakeEmbedder):
        failures = 1

        def encode_media(self, items: list[dict[str, object]]) -> np.ndarray:
            if any("audio" in item for item in items) and self.failures:
                self.failures -= 1
                raise OutOfMemoryError("CUDA out of memory")
            return super().encode_media(items)

    embedder = FlakyGpu()
    p = Project(media_root, media_settings, embedder)
    try:
        assert p.wait_idle(30)
        p.request_scan(force=True)
        assert p.wait_idle(30)
        audio, _ = p.search("loop", k=5, modality="audio")

        assert {r.path for r in audio} == {"sfx/loop.wav", "sfx/other.wav"}
        assert embedder.media_encoded == 5      # nothing re-embedded twice
    finally:
        p.close()


def test_project_media_settings_change_rebuilds(media_root: Path,
                                                media_settings: Settings) -> None:
    # Regression: raising a media limit (or MEDIA_VERSION) left old chunks in place.
    first = Project(media_root, media_settings, FakeEmbedder())
    assert first.wait_idle(30)
    first.close()

    again = FakeEmbedder()
    p = Project(media_root, replace(media_settings, max_pdf_pages=1), again)
    try:
        assert p.wait_idle(30)
        assert again.media_encoded == 4          # rebuilt: 1 image, 1 pdf page, 2 audio
    finally:
        p.close()


def test_format_search_shows_page_and_time_locations() -> None:
    data = {"fresh": True, "index": {"files": 2, "chunks": 2}, "results": [
        {"path": "docs/manual.pdf", "lines": [None, None], "page": 3, "symbol": "p.3",
         "kind": "page", "modality": "pdf", "score": 0.8, "preview": "install notes"},
        {"path": "clips/intro.mp4", "lines": [None, None], "t_start": 80.0, "t_end": 100.0,
         "symbol": "01:20-01:40", "kind": "segment", "modality": "video", "score": 0.7,
         "preview": "[video 01:20-01:40 of 05:00]"},
    ]}

    text = format_search("q", data)

    assert "1. docs/manual.pdf p.3  p.3  [page]" in text
    assert "2. clips/intro.mp4 01:20-01:40" in text
