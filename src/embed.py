"""
Generate embeddings for corpus and queries using sentence-transformers.

Supports:
- BAAI/bge-small-en-v1.5 (384-dim, local, fast)
- text-embedding-3-small (1536-dim, OpenAI API, $0.10)

Caches embeddings as .npy files for reuse.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os


def load_jsonl(path: str) -> list:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def embed_corpus(
    model_name: str,
    corpus_path: str,
    output_path: str,
    batch_size: int = 32,
) -> tuple:
    """
    Generate embeddings for corpus documents.

    Args:
        model_name: HuggingFace model name (e.g., "BAAI/bge-small-en-v1.5")
        corpus_path: Path to corpus.jsonl
        output_path: Path to save embeddings.npy
        batch_size: Batch size for processing

    Returns:
        (embeddings, doc_ids): embeddings shape (N, dim), doc_ids list
    """

    print(f"\n🔤 Loading corpus from {corpus_path}")
    corpus = load_jsonl(corpus_path)
    print(f"Loaded {len(corpus):,} documents")

    # Prepare texts: combine title and text
    texts = []
    doc_ids = []
    for doc in corpus:
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        # Combine title and text for better embeddings
        combined = f"{title} {text}".strip() if title else text
        texts.append(combined)
        doc_ids.append(doc["_id"])

    # Load model
    print(f"\n🤖 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")

    # Generate embeddings
    print(f"\n📊 Generating embeddings for {len(texts):,} documents...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize
    )

    # Ensure float32
    embeddings = embeddings.astype(np.float32)

    # Save embeddings
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"✓ Saved embeddings to {output_path}")

    # Save doc_ids for reference
    doc_ids_path = output_path.replace(".npy", "_doc_ids.json")
    with open(doc_ids_path, "w") as f:
        json.dump(doc_ids, f)
    print(f"✓ Saved doc_ids to {doc_ids_path}")

    return embeddings, doc_ids


def embed_queries(
    model_name: str,
    queries_path: str,
    output_path: str,
    batch_size: int = 32,
) -> tuple:
    """
    Generate embeddings for queries.

    Args:
        model_name: HuggingFace model name
        queries_path: Path to queries.jsonl
        output_path: Path to save query_embeddings.npy
        batch_size: Batch size for processing

    Returns:
        (embeddings, query_ids): embeddings shape (Q, dim), query_ids list
    """

    print(f"\n🔤 Loading queries from {queries_path}")
    queries = load_jsonl(queries_path)
    print(f"Loaded {len(queries):,} queries")

    texts = []
    query_ids = []
    for query in queries:
        texts.append(query["text"])
        query_ids.append(query["_id"])

    # Load model
    print(f"\n🤖 Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Generate embeddings
    print(f"\n📊 Generating embeddings for {len(texts):,} queries...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(np.float32)

    # Save embeddings
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"✓ Saved query embeddings to {output_path}")

    # Save query_ids for reference
    query_ids_path = output_path.replace(".npy", "_query_ids.json")
    with open(query_ids_path, "w") as f:
        json.dump(query_ids, f)
    print(f"✓ Saved query_ids to {query_ids_path}")

    return embeddings, query_ids


def load_qrels(qrels_path: str) -> dict:
    """
    Load qrels (relevance judgments).

    Args:
        qrels_path: Path to qrels.tsv

    Returns:
        dict: {query_id: {relevant_doc_id, ...}}
    """
    qrels = {}
    with open(qrels_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id, doc_id, relevance = parts[0], parts[1], int(parts[2])
                if relevance > 0:  # Only include relevant documents
                    if query_id not in qrels:
                        qrels[query_id] = set()
                    qrels[query_id].add(doc_id)

    print(f"\n✓ Loaded {len(qrels):,} queries with relevant documents")
    return qrels


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for corpus and queries")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Model name from HuggingFace (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="data/corpus.jsonl",
        help="Path to corpus.jsonl",
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default="data/queries.jsonl",
        help="Path to queries.jsonl",
    )
    parser.add_argument(
        "--qrels-path",
        type=str,
        default="data/qrels.tsv",
        help="Path to qrels.tsv",
    )
    parser.add_argument(
        "--corpus-embeddings",
        type=str,
        default="data/corpus_embeddings.npy",
        help="Output path for corpus embeddings",
    )
    parser.add_argument(
        "--query-embeddings",
        type=str,
        default="data/query_embeddings.npy",
        help="Output path for query embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )

    args = parser.parse_args()

    print("="*60)
    print("🚀 Embedding Generation")
    print("="*60)
    print(f"Model:              {args.model}")
    print(f"Batch size:         {args.batch_size}")
    print("="*60)

    # Generate corpus embeddings
    if not os.path.exists(args.corpus_embeddings):
        embed_corpus(
            args.model,
            args.corpus_path,
            args.corpus_embeddings,
            args.batch_size,
        )
    else:
        print(f"ℹ️  Corpus embeddings already exist: {args.corpus_embeddings}")

    # Generate query embeddings
    if not os.path.exists(args.query_embeddings):
        embed_queries(
            args.model,
            args.queries_path,
            args.query_embeddings,
            args.batch_size,
        )
    else:
        print(f"ℹ️  Query embeddings already exist: {args.query_embeddings}")

    # Load qrels
    qrels = load_qrels(args.qrels_path)

    print("\n" + "="*60)
    print("✅ Embedding generation complete!")
    print("="*60)
    print(f"Corpus embeddings:  {args.corpus_embeddings}")
    print(f"Query embeddings:   {args.query_embeddings}")
    print(f"Qrels:              {args.qrels_path}")
    print("="*60)


if __name__ == "__main__":
    main()
