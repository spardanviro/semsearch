"""Phase 0 spike: EmbeddingGemma 2 feasibility on this machine.

Checks:
  1. text-only vs full-model loading: VRAM, load time, and whether text vectors match
  2. encoding throughput on real code chunks (Python stdlib) at several batch sizes
  3. retrieval quality on stdlib with known answers (EN + ZH queries):
     prompt SearchQuery vs CodeRetrieval, dims 768 / 256 / 128
  4. multimodal smoke test (image) on the full model

Run: uv run python spikes/phase0_model.py
Writes spikes/results/phase0_model.json
"""
from __future__ import annotations

import ast
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
MAX_SEQ = 1024
OUT = Path(__file__).parent / "results" / "phase0_model.json"
STDLIB = Path(sys.base_prefix) / "Lib"
CORPUS_FILES = [
    "shutil.py", "urllib/parse.py", "json/encoder.py", "json/decoder.py",
    "logging/__init__.py", "logging/handlers.py", "subprocess.py", "fnmatch.py",
    "ntpath.py", "posixpath.py", "argparse.py", "http/client.py", "tempfile.py",
    "difflib.py", "pprint.py", "base64.py", "functools.py", "configparser.py",
    "zipfile/__init__.py", "csv.py", "textwrap.py", "string.py", "glob.py",
]

# (query, expected qualified names; a hit = any expected name in top-k)
QUERIES: list[tuple[str, list[str]]] = [
    ("copy a directory tree recursively", ["shutil.copytree", "shutil._copytree"]),
    ("delete a directory tree, handling errors on read-only files", ["shutil.rmtree", "shutil._rmtree_unsafe", "shutil._rmtree_safe_fd"]),
    ("find an executable program on PATH", ["shutil.which"]),
    ("parse a URL query string into a dictionary", ["parse.parse_qs", "parse.parse_qsl"]),
    ("percent-encode a string so it is safe in a URL", ["parse.quote", "parse.quote_from_bytes", "parse.quote_plus"]),
    ("escape special characters when serializing a string to JSON", ["encoder.py_encode_basestring", "encoder.py_encode_basestring_ascii"]),
    ("rotate the log file when it grows beyond a size limit", ["handlers.RotatingFileHandler.shouldRollover", "handlers.RotatingFileHandler.doRollover"]),
    ("run a command and kill it if it exceeds a timeout", ["subprocess.run", "subprocess.Popen.communicate", "subprocess.Popen.wait", "subprocess.call"]),
    ("convert a shell wildcard pattern into a regular expression", ["fnmatch.translate", "fnmatch._translate"]),
    ("compute a relative path from one directory to another", ["ntpath.relpath", "posixpath.relpath"]),
    ("print the usage and help message for command-line options", ["argparse.ArgumentParser.print_help", "argparse.ArgumentParser.format_help", "argparse.ArgumentParser.print_usage", "argparse.ArgumentParser.format_usage"]),
    ("read an HTTP response body sent with chunked transfer encoding", ["client.HTTPResponse._read_chunked", "client.HTTPResponse._get_chunk_left", "client.HTTPResponse._read_next_chunk_size"]),
    ("create a temporary file with a unique name securely", ["tempfile.mkstemp", "tempfile._mkstemp_inner", "tempfile.NamedTemporaryFile"]),
    ("show the differences between two lists of lines in unified diff format", ["difflib.unified_diff"]),
    ("pretty-print nested data structures with indentation", ["pprint.pformat", "pprint.pprint", "pprint.PrettyPrinter.pformat", "pprint.PrettyPrinter._format"]),
    ("decode base64 encoded data", ["base64.b64decode", "base64.standard_b64decode", "base64.urlsafe_b64decode"]),
    ("memoize function results with a least-recently-used cache", ["functools.lru_cache", "functools._lru_cache_wrapper"]),
    ("parse an INI style configuration file", ["configparser.RawConfigParser._read", "configparser.RawConfigParser.read", "configparser.RawConfigParser.read_file", "configparser.RawConfigParser._read_inner"]),
    ("wrap long text to a fixed line width", ["textwrap.wrap", "textwrap.fill", "textwrap.TextWrapper.wrap", "textwrap.TextWrapper._wrap_chunks"]),
    ("expand wildcard patterns to matching file paths recursively", ["glob.glob", "glob.iglob", "glob._iglob", "glob._rlistdir", "glob._glob2"]),
    # Chinese queries
    ("递归复制整个目录", ["shutil.copytree", "shutil._copytree"]),
    ("在 PATH 环境变量里查找可执行文件", ["shutil.which"]),
    ("把 URL 查询字符串解析成字典", ["parse.parse_qs", "parse.parse_qsl"]),
    ("日志文件超过大小上限时自动轮转", ["handlers.RotatingFileHandler.shouldRollover", "handlers.RotatingFileHandler.doRollover"]),
    ("命令执行超时后杀掉子进程", ["subprocess.run", "subprocess.Popen.communicate", "subprocess.Popen.wait", "subprocess.call"]),
    ("计算两个目录之间的相对路径", ["ntpath.relpath", "posixpath.relpath"]),
    ("生成统一格式的文本差异对比", ["difflib.unified_diff"]),
    ("解析 INI 配置文件", ["configparser.RawConfigParser._read", "configparser.RawConfigParser.read", "configparser.RawConfigParser.read_file", "configparser.RawConfigParser._read_inner"]),
    ("读取 zip 压缩包里的某个文件", ["__init__.ZipFile.open", "__init__.ZipFile.read"]),
    ("把长文本按固定宽度换行", ["textwrap.wrap", "textwrap.fill", "textwrap.TextWrapper.wrap", "textwrap.TextWrapper._wrap_chunks"]),
]


@dataclass(frozen=True)
class Chunk:
    qualname: str
    path: str
    start: int
    end: int
    text: str


def chunk_file(path: Path) -> list[Chunk]:
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    rel = path.relative_to(STDLIB).as_posix()
    mod = Path(rel).stem
    out: list[Chunk] = []

    def add(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, qual: str, kind: str) -> None:
        end = node.end_lineno or node.lineno
        body = "\n".join(lines[node.lineno - 1: end])
        header = f"path: Lib/{rel}\nsymbol: {qual}\nkind: {kind}\nlang: python\n---\n"
        out.append(Chunk(f"{mod}.{qual}", rel, node.lineno, end, header + body))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name, "function")
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if not methods:
                add(node, node.name, "class")
            for m in methods:
                add(m, f"{node.name}.{m.name}", "method")
    return out


def vram_mb() -> float:
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def load(text_only: bool) -> tuple[SentenceTransformer, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    config_kwargs = {"vision_config": None, "audio_config": None} if text_only else None
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, device="cuda", model_kwargs={"dtype": torch.bfloat16},
                                config_kwargs=config_kwargs)
    model.max_seq_length = MAX_SEQ
    return model, round(time.perf_counter() - t0, 1)


def unload(model: SentenceTransformer) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def encode_docs(model: SentenceTransformer, texts: list[str], batch: int) -> np.ndarray:
    return model.encode(texts, prompt_name="Document", batch_size=batch,
                        convert_to_numpy=True, normalize_embeddings=True)


def throughput(model: SentenceTransformer, texts: list[str]) -> list[dict]:
    rows = []
    sample = texts[:512]
    for bs in (8, 16, 32, 64):
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            encode_docs(model, sample, bs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rows.append({"batch": bs, "chunks_per_s": round(len(sample) / dt, 1), "peak_vram_mb": vram_mb()})
        except torch.OutOfMemoryError:
            rows.append({"batch": bs, "oom": True})
            torch.cuda.empty_cache()
    return rows


def evaluate(doc_emb: np.ndarray, chunks: list[Chunk], q_emb: np.ndarray) -> dict:
    names = [c.qualname for c in chunks]
    res = {}
    for dim in (768, 256, 128):
        d = doc_emb[:, :dim] / np.linalg.norm(doc_emb[:, :dim], axis=1, keepdims=True)
        q = q_emb[:, :dim] / np.linalg.norm(q_emb[:, :dim], axis=1, keepdims=True)
        scores = q @ d.T
        hits1 = hits5 = hits10 = 0
        rr = 0.0
        misses = []
        for i, (query, expected) in enumerate(QUERIES):
            order = np.argsort(-scores[i])[:10]
            ranked = [names[j] for j in order]
            rank = next((r for r, n in enumerate(ranked, 1) if n in expected), None)
            if rank:
                hits1 += rank == 1
                hits5 += rank <= 5
                hits10 += 1
                rr += 1 / rank
            else:
                misses.append({"q": query, "top3": ranked[:3]})
        n = len(QUERIES)
        res[dim] = {"R@1": round(hits1 / n, 3), "R@5": round(hits5 / n, 3),
                    "R@10": round(hits10 / n, 3), "MRR": round(rr / n, 3), "misses": misses}
    return res


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    chunks = [c for f in CORPUS_FILES for c in chunk_file(STDLIB / f)]
    names = {c.qualname for c in chunks}
    missing = sorted({e for _, exp in QUERIES for e in exp if e not in names} - set())
    texts = [c.text for c in chunks]
    queries = [q for q, _ in QUERIES]
    report: dict = {"corpus_chunks": len(chunks), "expected_names_not_in_corpus": missing}

    # --- text-only model
    model, load_s = load(text_only=True)
    text_only: dict[str, Any] = {"load_s": load_s, "vram_after_load_mb": vram_mb(),
                                 "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    report["text_only"] = text_only
    tok = model.tokenizer
    lens = [len(tok(t)["input_ids"]) for t in texts]
    report["chunk_tokens"] = {"mean": int(np.mean(lens)), "p50": int(np.median(lens)),
                              "p95": int(np.percentile(lens, 95)), "max": max(lens),
                              "over_1024": sum(x > MAX_SEQ for x in lens)}
    text_only["throughput"] = throughput(model, texts)

    t0 = time.perf_counter()
    doc_emb = encode_docs(model, texts, 16)
    text_only["full_corpus_encode_s"] = round(time.perf_counter() - t0, 1)

    report["retrieval"] = {}
    for prompt in ("SearchQuery", "CodeRetrieval"):
        q_emb = model.encode(queries, prompt_name=prompt, convert_to_numpy=True, normalize_embeddings=True)
        report["retrieval"][prompt] = evaluate(doc_emb, chunks, q_emb)

    probe = texts[:64] + queries
    probe_text_only = encode_docs(model, probe, 16)
    unload(model)

    # --- full multimodal model
    model, load_s = load(text_only=False)
    report["full"] = {"load_s": load_s, "vram_after_load_mb": vram_mb(),
                      "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    probe_full = encode_docs(model, probe, 16)
    cos = np.sum(probe_text_only * probe_full, axis=1)
    report["text_only_vs_full_cosine"] = {"min": float(cos.min()), "mean": float(cos.mean())}

    try:
        from PIL import Image
        img = Image.new("RGB", (448, 448), (30, 120, 200))
        torch.cuda.reset_peak_memory_stats()
        img_emb = model.encode([{"image": img}], convert_to_numpy=True, normalize_embeddings=True)
        txt = model.encode(["a solid blue square", "python source code for an HTTP client"],
                           prompt_name="SearchQuery", convert_to_numpy=True, normalize_embeddings=True)
        report["image_smoke"] = {"ok": True, "dim": int(img_emb.shape[1]), "peak_vram_mb": vram_mb(),
                                 "sim_blue_square": float(img_emb[0] @ txt[0]),
                                 "sim_http_client": float(img_emb[0] @ txt[1])}
    except Exception as exc:  # report, do not hide
        report["image_smoke"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    unload(model)

    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
