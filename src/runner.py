import time
import numpy as np
from typing import Dict, List

WARMUP_QUERIES = 50   # перші N запитів НЕ враховуємо (cold cache, JIT)
NUM_REPEATS = 3       # повторюємо вимір, беремо медіану


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