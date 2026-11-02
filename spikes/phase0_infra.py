"""Phase 0 spike: storage (sqlite-vec) and chunking (tree-sitter) on Windows.

Run: uv run python spikes/phase0_infra.py [path/to/sample.cs ...]
Writes spikes/results/phase0_infra.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sqlite_vec
from tree_sitter_language_pack import get_parser

OUT = Path(__file__).parent / "results" / "phase0_infra.json"
DIM = 768
N_VECS = 50_000

# node types that make a good chunk boundary, per language
DEF_TYPES = {
    "python": {"function_definition", "class_definition"},
    "csharp": {"method_declaration", "constructor_declaration", "property_declaration",
               "class_declaration", "struct_declaration", "interface_declaration",
               "enum_declaration"},
    "typescript": {"function_declaration", "method_definition", "class_declaration",
                   "interface_declaration", "lexical_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
}

SAMPLES = {
    "python": b"class A:\n    def f(self, x):\n        return x + 1\n\ndef g():\n    pass\n",
    "typescript": b"export class Foo { bar(x: number): number { return x } }\nfunction baz() {}\n",
    "go": b"package main\nfunc (s *S) M() {}\nfunc F() int { return 1 }\n",
    "csharp": b"namespace N { public class C { public int P { get; set; } public void M() {} } }",
}


def spike_sqlite_vec() -> dict:
    res: dict = {"sqlite_version": sqlite3.sqlite_version}
    with tempfile.TemporaryDirectory() as tmp:
        db = sqlite3.connect(Path(tmp) / "t.db")
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        res["vec_version"] = db.execute("select vec_version()").fetchone()[0]
        db.execute("pragma journal_mode=wal")
        db.execute(f"create virtual table v using vec0(chunk_id integer primary key, "
                   f"embedding float[{DIM}] distance_metric=cosine)")
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((N_VECS, DIM)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        t0 = time.perf_counter()
        with db:
            db.executemany("insert into v(chunk_id, embedding) values (?, ?)",
                           ((i, vecs[i].tobytes()) for i in range(N_VECS)))
        res["insert_50k_s"] = round(time.perf_counter() - t0, 2)

        q = vecs[123]
        rows: list[tuple[int, float]] = []
        t0 = time.perf_counter()
        for _ in range(20):
            rows = db.execute("select chunk_id, distance from v where embedding match ? "
                              "and k = 32 order by distance", (q.tobytes(),)).fetchall()
        res["knn_k32_ms"] = round((time.perf_counter() - t0) / 20 * 1000, 1)
        res["knn_self_hit_first"] = rows[0][0] == 123

        # numpy brute force for comparison
        t0 = time.perf_counter()
        for _ in range(20):
            np.argpartition(-(vecs @ q), 32)[:32]
        res["numpy_bruteforce_ms"] = round((time.perf_counter() - t0) / 20 * 1000, 1)

        t0 = time.perf_counter()
        with db:
            db.executemany("delete from v where chunk_id = ?", ((i,) for i in range(0, N_VECS, 10)))
            db.executemany("insert into v(chunk_id, embedding) values (?, ?)",
                           ((i, vecs[i].tobytes()) for i in range(0, N_VECS, 10)))
        res["delete_reinsert_5k_s"] = round(time.perf_counter() - t0, 2)
        res["count_after"] = db.execute("select count(*) from v").fetchone()[0]

        db.execute("create virtual table f using fts5(text, symbol, path)")
        db.execute("insert into f values ('def backoff(self): retry later', 'RetryPolicy.backoff', 'a.py')")
        res["fts5_ok"] = db.execute("select count(*) from f where f match 'retry'").fetchone()[0] == 1
        size_mb = (Path(tmp) / "t.db").stat().st_size / 2**20
        db.close()
    res["db_size_50k_mb"] = round(size_mb, 1)
    return res


def collect_defs(lang: str, src: bytes) -> list[tuple[str, int, int]]:
    tree = get_parser(lang).parse(src)
    wanted = DEF_TYPES[lang]
    out: list[tuple[str, int, int]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in wanted:
            out.append((node.type, node.start_point[0] + 1, node.end_point[0] + 1))
        stack.extend(reversed(node.children))
    return out


def spike_tree_sitter(cs_files: list[Path]) -> dict:
    res: dict = {"samples": {}}
    for lang, src in SAMPLES.items():
        try:
            res["samples"][lang] = collect_defs(lang, src)
        except Exception as exc:
            res["samples"][lang] = f"ERROR {type(exc).__name__}: {exc}"
    real = []
    for f in cs_files:
        src = f.read_bytes()
        t0 = time.perf_counter()
        defs = collect_defs("csharp", src)
        real.append({"file": f.name, "kb": round(len(src) / 1024, 1),
                     "lines": src.count(b"\n") + 1, "defs": len(defs),
                     "parse_ms": round((time.perf_counter() - t0) * 1000, 1),
                     "first": defs[:3]})
    res["real_csharp"] = real
    return res


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    cs_files = [Path(p) for p in sys.argv[1:]]
    report = {"sqlite_vec": spike_sqlite_vec(), "tree_sitter": spike_tree_sitter(cs_files)}
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
