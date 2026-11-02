"""Media files -> embeddable chunks (phase 2/3): images, PDF pages, audio windows, video segments.

`segment()` is cheap: it reads sizes, page texts and durations, never pixels or samples, so a
long video yields its chunk list without being decoded. `load()` decodes one chunk into the
input EmbeddingGemma 2 expects through sentence-transformers (spikes/phase2_media.py):
- image: a PIL image (no prompt)
- PDF page: "title: <file> p.N | text: <page text> <|image|>" + the rendered page
- audio: mono 16 kHz float32 samples (no prompt)
- video: frames sampled at 1 fps, plus the segment's audio when the audio encoder is loaded
Decoding uses PyAV / pypdfium2 / Pillow directly, so neither librosa nor torchcodec is needed.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from semsearch.config import Settings
from semsearch.formatting import clock

log = logging.getLogger(__name__)

AUDIO_RATE = 16_000
AUDIO_WINDOW_S = 30.0          # 25 tokens/s -> 750 tokens per window
AUDIO_HOP_S = 25.0             # 5 s overlap so a sound on a boundary is never split in two
VIDEO_SEGMENT_S = 20.0         # 1 fps x 140 tokens = 2800 tokens (+ 500 for its audio)
MIN_TAIL_S = 5.0               # a shorter last piece is merged into the previous one
MIN_IMAGE_SIDE = 32            # smaller images are icons/sprites: skipped
MAX_IMAGE_PIXELS = 64_000_000  # larger images (or decompression bombs) are skipped
IMAGE_MAX_SIDE = 1536          # the processor resizes anyway; keep decoded copies small
PDF_RENDER_SIDE = 1280         # longest side of a rendered page, in pixels
PDF_TEXT_CHARS = 3000          # page text embedded together with the page image
PREVIEW_CHARS = 300
VIDEO_FRAME_SIDE = 768


@dataclass(frozen=True)
class MediaChunk:
    kind: str                   # image | pdf | audio | video
    symbol: str                 # what results show as the "symbol": file name, "p.3", "01:20-01:40"
    preview: str
    text: str = ""              # searchable words (PDF page text)
    page: int | None = None     # 1-based PDF page
    t_start: float | None = None
    t_end: float | None = None

    @property
    def locator(self) -> str:
        """Stable identity inside the file (part of the chunk hash)."""
        return f"{self.kind}:{self.page}:{self.t_start}:{self.t_end}"



def windows(duration: float, length: float, hop: float) -> list[tuple[float, float]]:
    """Cover [0, duration] with windows; a tail shorter than MIN_TAIL_S joins the previous."""
    if duration <= 0:
        return []
    out: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + length, duration)
        if out and end - start < MIN_TAIL_S:
            out[-1] = (out[-1][0], end)
            break
        out.append((start, end))
        if end >= duration:
            break
        start += hop
    return out


# --- segmentation (cheap) --------------------------------------------------------------------
def segment(path: Path, kind: str, settings: Settings) -> list[MediaChunk]:
    if kind == "image":
        return _segment_image(path)
    if kind == "pdf":
        return _segment_pdf(path, settings)
    if kind in ("audio", "video"):
        return _segment_timed(path, kind, settings)
    raise ValueError(f"unknown media kind {kind!r}")


# Pillow detects formats by content, not extension: a crafted "cat.png" could be EPS, which
# Pillow hands to Ghostscript. Only the formats we index are ever opened.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "BMP", "GIF")


def _segment_image(path: Path) -> list[MediaChunk]:
    from PIL import Image
    with Image.open(path, formats=IMAGE_FORMATS) as img:     # reads the header only
        width, height = img.size
    if width < MIN_IMAGE_SIDE and height < MIN_IMAGE_SIDE:
        return []
    if width * height > MAX_IMAGE_PIXELS:    # decompression bomb or a giant scan: skip
        log.info("%s: %dx%d is too large to index", path.name, width, height)
        return []
    return [MediaChunk("image", path.name, f"[image {width}x{height}]")]


def _segment_pdf(path: Path, settings: Settings) -> list[MediaChunk]:
    import pypdfium2 as pdfium
    out: list[MediaChunk] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        pages = len(pdf)
        if pages > settings.max_pdf_pages:
            log.info("%s: indexing the first %d of %d pages", path.name, settings.max_pdf_pages, pages)
        for i in range(min(pages, settings.max_pdf_pages)):
            try:
                words = " ".join(pdf[i].get_textpage().get_text_range().split())
            except Exception as exc:         # one broken page must not drop the others
                log.info("%s p.%d: no text (%s)", path.name, i + 1, exc)
                words = ""
            preview = words[:PREVIEW_CHARS] + ("…" if len(words) > PREVIEW_CHARS else "")
            out.append(MediaChunk("pdf", f"p.{i + 1}", preview or f"[page {i + 1}, no text layer]",
                                  text=words[:PDF_TEXT_CHARS], page=i + 1))
    finally:
        pdf.close()
    return out


def _duration(container: Any) -> float:
    if container.duration:
        return float(container.duration) / 1_000_000          # av.time_base is microseconds
    for stream in container.streams:
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
    return 0.0


def _segment_timed(path: Path, kind: str, settings: Settings) -> list[MediaChunk]:
    import av
    with av.open(str(path)) as container:
        if kind == "audio" and not container.streams.audio:
            return []
        if kind == "video" and not container.streams.video:
            return []
        total = _duration(container)
    if total <= 0:
        raise ValueError(f"{path.name}: unknown duration")
    covered = min(total, settings.max_media_seconds)
    if covered < total:
        log.info("%s: indexing the first %s of %s", path.name, clock(covered), clock(total))
    length, hop = (AUDIO_WINDOW_S, AUDIO_HOP_S) if kind == "audio" else (VIDEO_SEGMENT_S,
                                                                         VIDEO_SEGMENT_S)
    out = []
    for start, end in windows(covered, length, hop):
        span = f"{clock(start)}-{clock(end)}"
        out.append(MediaChunk(kind, span, f"[{kind} {span} of {clock(total)}]",
                              t_start=round(start, 3), t_end=round(end, 3)))
    return out


# --- loading (decodes one chunk) -------------------------------------------------------------
def load(path: Path, chunk: MediaChunk, with_audio: bool) -> dict[str, Any]:
    """Model input for one chunk. `with_audio`: also give video segments their sound."""
    if chunk.kind == "image":
        return {"image": _load_image(path)}
    if chunk.kind == "pdf":
        return _load_pdf_page(path, chunk)
    assert chunk.t_start is not None and chunk.t_end is not None
    if chunk.kind == "audio":
        samples = _audio(path, chunk.t_start, chunk.t_end)
        if samples is None or not len(samples):
            raise ValueError(f"{path.name}: no audio in {chunk.symbol}")
        return {"audio": _audio_input(samples)}
    if chunk.kind == "video":
        frames = _frames(path, chunk.t_start, chunk.t_end)
        sound = _audio(path, chunk.t_start, chunk.t_end) if with_audio else None
        if sound is not None and len(sound) >= AUDIO_RATE:     # at least 1 s of real audio
            return {"text": "<|video|> <|audio|>", "video": frames, "audio": _audio_input(sound)}
        return {"video": frames}
    raise ValueError(f"unknown media kind {chunk.kind!r}")


def _audio_input(samples: np.ndarray) -> dict[str, Any]:
    return {"array": samples, "sampling_rate": AUDIO_RATE}


def _load_image(path: Path) -> Any:
    from PIL import Image
    with Image.open(path, formats=IMAGE_FORMATS) as img:
        img.seek(0)                                    # first frame of a GIF
        img.draft("RGB", (IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))   # JPEG: decode already downscaled
        if img.mode in ("P", "LA") or "transparency" in img.info:
            img = img.convert("RGBA")                  # palette transparency -> alpha first
        rgb = img.convert("RGB")
    rgb.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    return rgb


_pdfs = threading.local()      # one open PDF per indexing thread: pages of a file come together


def _open_pdf(path: Path) -> Any:
    import pypdfium2 as pdfium
    cached = getattr(_pdfs, "doc", None)
    if cached is not None and cached[0] == path:
        return cached[1]
    release()
    doc = pdfium.PdfDocument(str(path))
    _pdfs.doc = (path, doc)
    return doc


def release() -> None:
    """Close the cached PDF (Windows would otherwise keep the file locked)."""
    cached = getattr(_pdfs, "doc", None)
    _pdfs.doc = None
    if cached is not None:
        cached[1].close()


def _load_pdf_page(path: Path, chunk: MediaChunk) -> dict[str, Any]:
    assert chunk.page is not None
    page = _open_pdf(path)[chunk.page - 1]
    width, height = page.get_size()
    scale = PDF_RENDER_SIDE / max(width, height, 1.0)
    image = page.render(scale=scale).to_pil().convert("RGB")
    if not chunk.text:
        return {"image": image}
    return {"text": f"title: {path.name} {chunk.symbol} | text: {chunk.text} <|image|>",
            "image": [image]}


def _origin(stream: Any, container: Any) -> float:
    """Timestamp of t=0. mkv/webm/ts recordings and edit-listed mp4 often start far from 0;
    windows count from the start of the media, packet timestamps do not."""
    if stream.start_time is not None and stream.time_base:
        return float(stream.start_time * stream.time_base)
    if container.start_time is not None:
        return float(container.start_time) / 1_000_000
    return 0.0


def _audio(path: Path, start: float, end: float) -> np.ndarray | None:
    """Mono 16 kHz float32 samples of [start, end); None if the file has no audio stream."""
    import av
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        origin = _origin(stream, container)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_RATE)
        if start > 0 and stream.time_base:
            container.seek(int((origin + start) / stream.time_base), stream=stream)
        chunks: list[np.ndarray] = []
        have, skip = 0, 0
        need = int((end - start) * AUDIO_RATE)
        first: float | None = None
        for frame in container.decode(stream):
            if first is None:     # seeks land on a packet boundary at or before the target
                first = (float(frame.pts * stream.time_base) - origin
                         if frame.pts is not None else start)
                skip = max(0, int((start - first) * AUDIO_RATE))
            for out in resampler.resample(frame):
                block = out.to_ndarray().reshape(-1)
                chunks.append(block)
                have += len(block)
            if have >= skip + need:
                break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[skip:skip + need].astype(np.float32)


def _frames(path: Path, start: float, end: float) -> np.ndarray:
    """Frames at 1 fps in [start, end), longest side VIDEO_FRAME_SIDE, as (T, H, W, 3) uint8."""
    import av
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = _origin(stream, container)
        rate = float(stream.average_rate or stream.guessed_rate or 0)
        if start > 0 and stream.time_base:
            container.seek(int((origin + start) / stream.time_base), stream=stream)
        next_t, decoded = start, 0
        for frame in container.decode(stream):
            if frame.pts is not None and stream.time_base:
                t = float(frame.pts * stream.time_base) - origin
            elif rate:            # no timestamps: count frames from where decoding began
                t = start + decoded / rate
            else:
                t = next_t
            decoded += 1
            if t < next_t:
                continue
            if t >= end:
                break
            frames.append(_frame_rgb(frame))
            next_t += 1.0
    if not frames:
        raise ValueError(f"{path.name}: no frames in {clock(start)}-{clock(end)}")
    # frames of one video share a size; stack needs equal shapes (variable-size streams exist)
    shape = frames[0].shape
    return np.stack([f if f.shape == shape else np.asarray(
        _resize(f, shape[1], shape[0])) for f in frames])


def _frame_rgb(frame: Any) -> np.ndarray:
    """Scale in the decoder before converting: an 8K frame never exists as a full RGB copy."""
    side = max(frame.width, frame.height)
    if side > VIDEO_FRAME_SIDE:
        scale = VIDEO_FRAME_SIDE / side
        width = max(2, int(frame.width * scale) // 2 * 2)
        height = max(2, int(frame.height * scale) // 2 * 2)
        return frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
    return frame.to_ndarray(format="rgb24")


def _resize(frame: np.ndarray, width: int, height: int) -> Any:
    from PIL import Image
    return Image.fromarray(frame).resize((width, height))
