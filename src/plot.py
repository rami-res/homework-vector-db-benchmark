"""
Generate benchmark visualization plots from results.

Plots:
- Pareto frontier (recall vs latency)
- Latency distribution (p50/p95/p99)
- Disk size comparison
- Index time comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results CSV file."""
    return pd.read_csv(csv_path)


def plot_pareto_frontier(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot recall vs latency (Pareto frontier).

    Better databases are on the upper-right (high recall, low latency).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        "FAISS Flat": "#1f77b4",
        "FAISS HNSW": "#ff7f0e",
        "Qdrant": "#2ca02c",
        "pgvector": "#d62728",
        "Chroma": "#9467bd",
    }

    for db_name in df["database"].unique():
        db_data = df[df["database"] == db_name]
        recall = db_data["recall_at_10"].values[0]
        latency = db_data["latency_p50_ms"].values[0]
        color = colors.get(db_name, "#000000")

        ax.scatter(
            latency,
            recall,
            s=300,
            alpha=0.7,
            label=db_name,
            color=color,
            edgecolors="black",
            linewidth=2,
        )

        # Add label
        ax.annotate(
            db_name,
            xy=(latency, recall),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Latency (p50) [ms]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall@10", fontsize=12, fontweight="bold")
    ax.set_title("Vector DB Benchmark: Pareto Frontier (Recall vs Latency)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    recall_min = df["recall_at_10"].min()
    ax.set_ylim([max(0, recall_min - 0.05), 1.01])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved Pareto frontier plot to {output_path}")
    plt.close()


def plot_latency_distribution(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot latency percentiles (p50, p95, p99) for each database.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.25

    p50 = df["latency_p50_ms"].values
    p95 = df["latency_p95_ms"].values
    p99 = df["latency_p99_ms"].values

    bars1 = ax.bar(x - width, p50, width, label="p50", alpha=0.8, color="#1f77b4")
    bars2 = ax.bar(x, p95, width, label="p95", alpha=0.8, color="#ff7f0e")
    bars3 = ax.bar(x + width, p99, width, label="p99", alpha=0.8, color="#d62728")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Database", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency [ms]", fontsize=12, fontweight="bold")
    ax.set_title("Query Latency Percentiles (p50/p95/p99)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["database"].values, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved latency distribution plot to {output_path}")
    plt.close()


def plot_disk_size(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot disk size for each database.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    databases = df["database"].values
    disk_sizes = df["disk_mb"].values

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(databases, disk_sizes, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, size in zip(bars, disk_sizes):
        height = bar.get_height()
        label = f"{size:.1f} MB" if size > 0 else "0 MB"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Disk Size [MB]", fontsize=12, fontweight="bold")
    ax.set_title("Index Size on Disk", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved disk size plot to {output_path}")
    plt.close()


def plot_index_time(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot indexing time for each database.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    databases = df["database"].values
    index_times = df["index_time_sec"].values

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(databases, index_times, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, time in zip(bars, index_times):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{time:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Time [seconds]", fontsize=12, fontweight="bold")
    ax.set_title("Index Construction Time", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved index time plot to {output_path}")
    plt.close()


def plot_recall_vs_indexing(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot recall vs indexing time (trade-off analysis).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        "FAISS Flat": "#1f77b4",
        "FAISS HNSW": "#ff7f0e",
        "Qdrant": "#2ca02c",
        "pgvector": "#d62728",
        "Chroma": "#9467bd",
    }

    for idx, row in df.iterrows():
        db_name = row["database"]
        recall = row["recall_at_10"]
        index_time = row["index_time_sec"]
        color = colors.get(db_name, "#000000")

        ax.scatter(
            index_time,
            recall,
            s=300,
            alpha=0.7,
            label=db_name,
            color=color,
            edgecolors="black",
            linewidth=2,
        )

        ax.annotate(
            db_name,
            xy=(index_time, recall),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Index Time [seconds]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall@10", fontsize=12, fontweight="bold")
    ax.set_title("Recall vs Indexing Time Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved recall vs indexing plot to {output_path}")
    plt.close()


def print_summary_table(df: pd.DataFrame) -> None:
    """Print summary table to stdout."""
    print("\n" + "="*100)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("="*100)

    summary = df[["database", "recall_at_10", "mrr_at_10", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "index_time_sec", "disk_mb"]].copy()

    # Format float columns
    for col in ["recall_at_10", "mrr_at_10", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]:
        summary[col] = summary[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    for col in ["index_time_sec", "disk_mb"]:
        summary[col] = summary[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    print(summary.to_string(index=False))
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument(
        "--input",
        type=str,
        default="results/results.csv",
        help="Input CSV file with benchmark results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"📂 Loading results from {args.input}")
    df = load_results(args.input)

    # Print summary table
    print_summary_table(df)

    # Generate plots
    print("\n🎨 Generating plots...")
    plot_pareto_frontier(df, f"{args.output}/pareto_frontier.png")
    plot_latency_distribution(df, f"{args.output}/latency_distribution.png")
    plot_disk_size(df, f"{args.output}/disk_size.png")
    plot_index_time(df, f"{args.output}/index_time.png")
    plot_recall_vs_indexing(df, f"{args.output}/recall_vs_indexing.png")

    print("\n✅ All plots generated successfully!")
    print(f"📁 Output directory: {args.output}")


if __name__ == "__main__":
    main()
