"""
Metrics for evaluating vector database search quality.

Implemented metrics:
- Recall@K: proportion of relevant documents in top-K results
- MRR@K: Mean Reciprocal Rank (position of first relevant result)
- NDCG@K: Normalized Discounted Cumulative Gain
"""

from typing import List, Set


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@K.

    Recall@K = |retrieved[:K] ∩ relevant| / min(K, |relevant|)

    Args:
        retrieved: list of retrieved document IDs (in rank order)
        relevant: set of relevant document IDs
        k: number of top results to consider

    Returns:
        Recall value in [0, 1]
    """
    if not relevant:
        return 0.0

    hits = len(set(retrieved[:k]) & relevant)
    return hits / min(k, len(relevant))


def mrr_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Mean Reciprocal Rank@K.

    MRR@K = 1 / rank_of_first_relevant_doc (0 if not in top-K)

    Args:
        retrieved: list of retrieved document IDs (in rank order)
        relevant: set of relevant document IDs
        k: number of top results to consider

    Returns:
        MRR value in [0, 1]
    """
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K.

    NDCG@K = DCG@K / IDCG@K
    where:
      DCG@K = sum(relevance_i / log2(i+1)) for i in top-K
      IDCG@K = DCG of ideal ranking (all relevant docs first)

    Args:
        retrieved: list of retrieved document IDs (in rank order)
        relevant: set of relevant document IDs
        k: number of top results to consider

    Returns:
        NDCG value in [0, 1]
    """
    import math

    if not relevant:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            # Relevance = 1 for binary relevance
            dcg += 1.0 / math.log2(i + 2)

    # Calculate IDCG (ideal ranking)
    idcg = 0.0
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@K.

    Precision@K = |retrieved[:K] ∩ relevant| / K

    Args:
        retrieved: list of retrieved document IDs (in rank order)
        relevant: set of relevant document IDs
        k: number of top results to consider

    Returns:
        Precision value in [0, 1]
    """
    if k == 0:
        return 0.0

    hits = len(set(retrieved[:k]) & relevant)
    return hits / k


def mean_avg_precision(
    all_retrieved: List[List[str]],
    all_relevant: List[Set[str]],
    k: int,
) -> float:
    """
    Calculate Mean Average Precision@K across multiple queries.

    Args:
        all_retrieved: list of retrieved document lists (one per query)
        all_relevant: list of relevant document sets (one per query)
        k: number of top results to consider

    Returns:
        MAP value in [0, 1]
    """
    if not all_retrieved:
        return 0.0

    aps = []
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        # Calculate AP: average of precision at each relevant position
        ap = 0.0
        num_hits = 0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                num_hits += 1
                ap += num_hits / (i + 1)

        if relevant:
            ap /= min(k, len(relevant))
        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0
