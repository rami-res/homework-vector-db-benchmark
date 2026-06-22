"""
Load BeIR/quora dataset for benchmarking.

Downloads:
- ~523K documents (duplicate-pair questions)
- ~10K test queries
- Ground truth relevance labels (qrels)
"""

import json
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def load_beir_quora(output_dir: str = "data"):
    """
    Load BeIR/quora dataset from Hugging Face.

    Args:
        output_dir: Directory to save corpus, queries, and qrels

    Saves:
        - data/corpus.jsonl: {"_id": str, "title": str, "text": str}
        - data/queries.jsonl: {"_id": str, "text": str}
        - data/qrels.tsv: query_id \t doc_id \t relevance
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    queries_path = os.path.join(output_dir, "queries.jsonl")
    qrels_path = os.path.join(output_dir, "qrels.tsv")

    print("Loading BeIR/quora dataset...")

    # Load corpus
    print("\n📚 Loading corpus...")
    corpus = load_dataset("BeIR/quora", "corpus", split="corpus")

    with open(corpus_path, "w") as f:
        for doc in tqdm(corpus, desc="Writing corpus"):
            record = {
                "_id": doc["_id"],
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"✓ Saved {len(corpus):,} documents to {corpus_path}")

    # Load queries
    print("\n🔍 Loading queries...")
    queries = load_dataset("BeIR/quora", "queries", split="queries")

    with open(queries_path, "w") as f:
        for query in tqdm(queries, desc="Writing queries"):
            record = {
                "_id": query["_id"],
                "text": query.get("text", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"✓ Saved {len(queries):,} queries to {queries_path}")

    # Load qrels (relevance judgments) from BeIR/quora-qrels on HuggingFace
    print("\n✅ Loading qrels (ground truth)...")
    qrels_data = load_dataset("BeIR/quora-qrels", split="test")

    with open(qrels_path, "w") as f:
        for rel in tqdm(qrels_data, desc="Writing qrels"):
            f.write(f"{rel['query-id']}\t{rel['corpus-id']}\t{rel['score']}\n")

    print(f"✓ Saved {len(qrels_data):,} relevance judgments to {qrels_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("📊 Dataset Summary")
    print("="*60)
    print(f"Corpus size:      {len(corpus):>10,} documents")
    print(f"Queries:          {len(queries):>10,} queries")
    print(f"Qrels (test):     {len(qrels_data):>10,} relevance pairs")
    print(f"Corpus path:      {corpus_path}")
    print(f"Queries path:     {queries_path}")
    print(f"Qrels path:       {qrels_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load BeIR/quora dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save dataset files (default: data/)",
    )

    args = parser.parse_args()
    load_beir_quora(args.output_dir)
