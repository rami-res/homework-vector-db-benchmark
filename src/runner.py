import time
import numpy as np
import argparse
import json
import csv
import os
from typing import Dict, List
from pathlib import Path

from benchmarks.faiss_flat import FAISSFlat
from benchmarks.faiss_hnsw import FAISSHNSw
from benchmarks.qdrant_db import QdrantDB
from benchmarks.pgvector_db import PgvectorDB
from benchmarks.chroma_db import ChromaDB

WARMUP_QUERIES = 50   # перші N запитів НЕ враховуємо (cold cache, JIT)
NUM_REPEATS = 3       # повторюємо вимір, беремо медіану
TOP_K = 10            # число результатів для оцінки


def _recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Recall@K = |retrieved ∩ relevant| / min(K, |relevant|)."""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / min(k, len(relevant))


def _mrr_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """MRR@K = 1 / rank першого правильного результату (0 якщо нема)."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def benchmark_db(
    db,
    doc_vectors: np.ndarray,        # (N, dim) float32
    doc_ids: List[str],             # parallel ID's до doc_vectors
    query_vectors: np.ndarray,      # (Q, dim) float32
    query_ids: List[str],           # ID кожного запиту з qrels
    qrels: Dict[str, set],          # {query_id: {relevant_doc_id, ...}}
    top_k: int = 10,
) -> Dict:
    # === INDEX ===
    t0 = time.perf_counter()
    db.index(doc_vectors, ids=doc_ids)
    index_time = time.perf_counter() - t0

    # === WARMUP ===
    for q_vec in query_vectors[:WARMUP_QUERIES]:
        db.search(q_vec, top_k=top_k)

    # === MEASURED QUERIES (3 repeats, median) ===
    all_latencies: List[List[float]] = []
    recalls: List[float] = []
    mrrs: List[float] = []

    for repeat in range(NUM_REPEATS):
        latencies = []
        for q_vec, q_id in zip(query_vectors, query_ids):
            t0 = time.perf_counter()
            results = db.search(q_vec, top_k=top_k)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

            if repeat == 0:
                retrieved_ids = [doc_id for doc_id, _score in results]
                relevant = qrels.get(q_id, set())
                recalls.append(_recall_at_k(retrieved_ids, relevant, top_k))
                mrrs.append(_mrr_at_k(retrieved_ids, relevant, top_k))
        all_latencies.append(latencies)

    # median across repeats per query, тоді percentiles
    latencies_arr = np.median(np.array(all_latencies), axis=0)

    return {
        "index_time_sec": round(index_time, 2),
        "disk_mb": round(db.disk_size_mb(), 1),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "mrr_at_10": round(float(np.mean(mrrs)), 4),
        "num_queries": len(query_vectors),
    }


def load_data(corpus_emb_path: str, query_emb_path: str, qrels_path: str):
    """Load corpus embeddings, query embeddings, and qrels."""
    print("\n📚 Loading data...")

    # Load corpus embeddings and doc IDs
    doc_embeddings = np.load(corpus_emb_path)
    doc_ids_path = corpus_emb_path.replace(".npy", "_doc_ids.json")
    with open(doc_ids_path, "r") as f:
        doc_ids = json.load(f)

    print(f"  ✓ Loaded {len(doc_embeddings):,} document embeddings ({doc_embeddings.shape[1]}-dim)")

    # Load query embeddings and query IDs
    query_embeddings = np.load(query_emb_path)
    query_ids_path = query_emb_path.replace(".npy", "_query_ids.json")
    with open(query_ids_path, "r") as f:
        query_ids = json.load(f)

    print(f"  ✓ Loaded {len(query_embeddings):,} query embeddings")

    # Load qrels (relevance judgments)
    qrels = {}
    with open(qrels_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                q_id, doc_id, relevance = parts[0], parts[1], int(parts[2])
                if relevance > 0:  # Only include relevant documents
                    if q_id not in qrels:
                        qrels[q_id] = set()
                    qrels[q_id].add(doc_id)

    print(f"  ✓ Loaded qrels for {len(qrels):,} queries")

    # Filter to only queries that have ground truth — training queries have no qrels
    # and would contribute recall=0, artificially lowering the average
    mask = [i for i, qid in enumerate(query_ids) if qid in qrels]
    query_embeddings = query_embeddings[mask]
    query_ids = [query_ids[i] for i in mask]
    print(f"  ✓ Filtered to {len(query_ids):,} queries with ground truth")

    return doc_embeddings, doc_ids, query_embeddings, query_ids, qrels


def run_all_benchmarks(
    corpus_emb_path: str,
    query_emb_path: str,
    qrels_path: str,
    output_csv: str,
    skip: List[str] = [],
) -> None:
    """Run benchmarks for all vector databases."""

    # Load data
    doc_embeddings, doc_ids, query_embeddings, query_ids, qrels = load_data(
        corpus_emb_path, query_emb_path, qrels_path
    )

    # Initialize all databases
    databases = {
        "FAISS Flat": FAISSFlat(output_path="/tmp/faiss_flat.index"),
        "FAISS HNSW": FAISSHNSw(output_path="/tmp/faiss_hnsw.index"),
        "Qdrant": QdrantDB(),
        "pgvector": PgvectorDB(),
        "Chroma": ChromaDB(persist_path="/tmp/chroma_data"),
    }

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "database",
        "index_time_sec",
        "disk_mb",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "recall_at_10",
        "mrr_at_10",
        "num_queries",
    ]

    def save_result(result: dict) -> None:
        """Merge one result into the CSV immediately — safe to call after each DB."""
        existing = {}
        if os.path.exists(output_csv):
            with open(output_csv, newline="") as f:
                for row in csv.DictReader(f):
                    existing[row["database"]] = row
        existing[result["database"]] = result
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing.values())
        print(f"  ✓ Saved to {output_csv}")

    results = []

    print("\n" + "="*80)
    print("🚀 STARTING BENCHMARKS")
    print("="*80)

    for db_name, db_instance in databases.items():
        if db_name in skip:
            print(f"\n⏭️  Skipping {db_name}")
            continue
        print(f"\n⏱️  Benchmarking {db_name}...")
        try:
            benchmark_result = benchmark_db(
                db_instance,
                doc_embeddings,
                doc_ids,
                query_embeddings,
                query_ids,
                qrels,
                top_k=TOP_K,
            )
            benchmark_result["database"] = db_name
            results.append(benchmark_result)
            save_result(benchmark_result)
            db_instance.cleanup()
            print(f"✅ {db_name} complete!\n")

        except Exception as e:
            print(f"❌ Error benchmarking {db_name}: {e}\n")
            try:
                db_instance.cleanup()
            except Exception:
                pass


    # Print summary — read from CSV so it always shows all accumulated results
    all_results = {}
    if os.path.exists(output_csv):
        with open(output_csv, newline="") as f:
            for row in csv.DictReader(f):
                all_results[row["database"]] = row

    print("="*80)
    print("📈 BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Database':<15} | {'Recall@10':>10} | {'MRR@10':>8} | {'p50 ms':>8} | {'p95 ms':>8} | {'p99 ms':>8} | {'Index s':>8} | {'Disk MB':>8}")
    print("-"*80)
    for row in all_results.values():
        print(f"{row['database']:<15} | "
              f"{float(row['recall_at_10']):>10.4f} | "
              f"{float(row['mrr_at_10']):>8.4f} | "
              f"{float(row['latency_p50_ms']):>8.3f} | "
              f"{float(row['latency_p95_ms']):>8.3f} | "
              f"{float(row['latency_p99_ms']):>8.3f} | "
              f"{float(row['index_time_sec']):>8.2f} | "
              f"{float(row['disk_mb']):>8.1f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run vector database benchmarks")
    parser.add_argument(
        "--corpus-embeddings",
        type=str,
        default="data/corpus_embeddings.npy",
        help="Path to corpus embeddings file",
    )
    parser.add_argument(
        "--query-embeddings",
        type=str,
        default="data/query_embeddings.npy",
        help="Path to query embeddings file",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default="data/qrels.tsv",
        help="Path to qrels file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        metavar="DB",
        help='Database names to skip, e.g. --skip "FAISS Flat" "Qdrant"',
    )

    args = parser.parse_args()

    # Check if files exist
    for file_path in [args.corpus_embeddings, args.query_embeddings, args.qrels]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found. Please run load_data.py and embed.py first.")
            return

    run_all_benchmarks(
        args.corpus_embeddings,
        args.query_embeddings,
        args.qrels,
        args.output,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()