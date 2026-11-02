"""Phase 2/3 spike: which input forms does EmbeddingGemma 2 (full load) accept through
sentence-transformers for images, PDF pages, audio and video, and do text queries find them?

Run: uv run python spikes/phase2_media.py <media dir>
The directory needs: images/*.png|jpg (2+), docs/*.pdf, audio/*.wav (2), video/*.webm|mp4 (1+).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from semsearch.index.media import AUDIO_RATE, _audio, _frames

OUT = Path(__file__).parent / "results" / "phase2_media.json"
results: dict[str, object] = {}


def try_encode(model: SentenceTransformer, label: str, item: object) -> np.ndarray | None:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        vec = model.encode([item], convert_to_numpy=True, normalize_embeddings=True)[0]
    except Exception as exc:  # we want to see every failure mode
        results[label] = f"FAIL {type(exc).__name__}: {exc}"[:300]
        print(f"[{label}] FAIL {type(exc).__name__}: {str(exc)[:200]}")
        traceback.print_exc(limit=1)
        return None
    ms = (time.perf_counter() - t0) * 1000
    peak = torch.cuda.max_memory_allocated() / 2**30
    results[label] = f"ok {ms:.0f} ms peak {peak:.2f} GB"
    print(f"[{label}] ok {ms:.0f} ms, peak {peak:.2f} GB")
    return vec


def main(root: Path) -> None:
    t0 = time.perf_counter()
    model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                model_kwargs={"dtype": torch.bfloat16})
    results["load_s"] = round(time.perf_counter() - t0, 1)
    results["load_vram_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    print(f"loaded full model in {results['load_s']}s, {results['load_vram_gb']} GB")

    def q(text: str) -> np.ndarray:
        return model.encode([text], prompt_name="SearchQuery", convert_to_numpy=True,
                            normalize_embeddings=True)[0]

    # images: path string vs PIL
    images = sorted([*root.glob("images/*.png"), *root.glob("images/*.jpg")])[:8]
    try_encode(model, "image:path-str", {"image": str(images[0])})
    for p in images:
        try_encode(model, f"image:pil:{p.name}", {"image": Image.open(p).convert("RGB")})

    # PDF page: interleaved text + page image vs image only
    pdf = pdfium.PdfDocument(str(next(root.glob("docs/*.pdf"))))
    page = pdf[0]
    text = page.get_textpage().get_text_range()[:1500]
    pil = page.render(scale=1.5).to_pil()
    v_mix = try_encode(model, "pdf:interleaved",
                       {"text": f"title: page 1 | text: {text} <|image|>", "image": [pil]})
    v_img = try_encode(model, "pdf:image-only", {"image": pil})
    if v_mix is not None and v_img is not None:
        qv = q("voxel world generation research paper")
        results["pdf_scores"] = {"interleaved": float(qv @ v_mix), "image_only": float(qv @ v_img)}

    # audio: ndarray / dict / path (path needs librosa: we decode with PyAV instead)
    wavs = sorted(root.glob("audio/*.wav"))
    arr = _audio(wavs[0], 0.0, 20.0)
    assert arr is not None
    try_encode(model, "audio:ndarray", {"audio": arr})
    try_encode(model, "audio:dict", {"audio": {"array": arr, "sampling_rate": AUDIO_RATE}})
    try_encode(model, "audio:path", {"audio": str(wavs[0])})

    # video: frames ndarray / list / path (path needs torchcodec), and frames + audio
    clip = next(iter(sorted([*root.glob("video/*.webm"), *root.glob("video/*.mp4")])))
    frames = _frames(clip, 0.0, 20.0)
    results["video_frames"] = list(frames.shape)
    try_encode(model, "video:frames-ndarray", {"video": frames})
    try_encode(model, "video:frames-list", {"video": list(frames)})
    try_encode(model, "video:path", {"video": str(clip)})
    try_encode(model, "video+audio:interleaved",
               {"text": "<|video|> <|audio|>", "video": frames,
                "audio": {"array": arr, "sampling_rate": AUDIO_RATE}})

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
