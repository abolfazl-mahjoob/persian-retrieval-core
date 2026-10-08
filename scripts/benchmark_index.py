"""Repeatable local in-memory BM25 timing; NOT a cross-machine performance claim."""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
import tracemalloc

from persian_retrieval import LexicalIndex

VOCAB = (
    "دانشجو", "آموزش", "دوره", "وبینار", "پرداخت", "موجودی", "پشتیبانی",
    "درگاه", "ثبت‌نام", "محصول", "سفارش", "مرجوعی", "کلاس", "قیمت",
    "کاربر", "سرور", "توسعه", "سرویس", "پایتون", "فاکتور", "ارسال",
    "Python", "React", "API", "Docker", "database", "security",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if not 1 <= args.documents <= 100_000 or not 1 <= args.queries <= 20_000:
        parser.error("outside safe benchmark bounds")
    randomizer = random.Random(args.seed)
    documents = {
        str(i): " ".join(randomizer.choices(VOCAB, k=randomizer.randrange(20, 80)))
        for i in range(args.documents)
    }
    queries = [
        " ".join(randomizer.sample(VOCAB, 3))
        for _ in range(args.queries)
    ]
    tracemalloc.start()
    start = time.perf_counter()
    index = LexicalIndex(documents)
    build_ms = (time.perf_counter() - start) * 1000.0
    _, index_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Measure query execution outside Python allocation tracing to avoid altering timing.
    for query in queries[:5]:
        index.search(query)
    latencies: list[float] = []
    hits = 0
    for query in queries:
        before = time.perf_counter_ns()
        result = index.search(query)
        latencies.append((time.perf_counter_ns() - before) / 1e6)
        hits += len(result)

    print(json.dumps({
        "type": "local_synthetic_microbenchmark",
        "hardware": platform.platform(),
        "python": platform.python_version(),
        "seed": args.seed,
        "documents": args.documents,
        "queries": args.queries,
        "vocabulary_size": index.vocabulary_size,
        "build_ms": round(build_ms, 2),
        "peak_traced_bytes_during_build": index_peak_bytes,
        "mean_query_ms": round(statistics.mean(latencies), 3),
        "p50_query_ms": round(percentile(latencies, 0.50), 3),
        "p95_query_ms": round(percentile(latencies, 0.95), 3),
        "result_count": hits,
        "limitations": [
            "Synthetic and seed-controlled documents",
            "Peak allocation traces index construction only; input corpus was allocated before tracing",
            "Not comparable across machines without environment reporting",
            "No durability, concurrency or cross-process benchmark",
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
