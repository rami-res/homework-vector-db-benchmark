# 🧪 Vector Database Benchmark Lab

This project benchmarks 5 major vector databases on the [BeIR/quora](https://huggingface.co/datasets/BeIR/quora) dataset (~523K documents, ~10K queries).

## 📊 Databases Benchmarked

| Database | Type | Recall | Speed | Key Metric |
|----------|------|--------|-------|-----------|
| **FAISS Flat** | In-memory | 100% | Baseline | Exact nearest neighbors |
| **FAISS HNSW** | In-memory | ~95-99% | Fast | Approximate with HNSW |
| **Qdrant** | Docker service | ~95-98% | Fast | Native vector DB |
| **pgvector** | PostgreSQL | ~95-98% | Fast | SQL-based with HNSW |
| **Chroma** | Embedded | ~95-99% | Fast | No external services |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
cd homework-vector-db-benchmark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Docker Services (Qdrant + PostgreSQL)

```bash
# Start Docker services
docker compose up -d

# Verify services are running
docker compose ps

# Expected output:
# qdrant       - listening on http://localhost:6333
# postgres     - listening on localhost:5432
```

### 3. Load Dataset

```bash
# Download BeIR/quora dataset (~5 minutes)
python src/load_data.py

# Output:
# ✓ data/corpus.jsonl       (~523K documents)
# ✓ data/queries.jsonl      (~10K queries)
# ✓ data/qrels.tsv          (ground truth)
```

### 4. Generate Embeddings

```bash
# Generate embeddings using BAAI/bge-small-en-v1.5 (~20-30 min on CPU)
python src/embed.py \
    --model BAAI/bge-small-en-v1.5 \
    --corpus-path data/corpus.jsonl \
    --queries-path data/queries.jsonl \
    --qrels-path data/qrels.tsv \
    --corpus-embeddings data/corpus_embeddings.npy \
    --query-embeddings data/query_embeddings.npy \
    --batch-size 32

# Output:
# ✓ data/corpus_embeddings.npy      (384-dim, ~2.1 GB)
# ✓ data/query_embeddings.npy       (384-dim, ~15 MB)
```

### 5. Run Benchmarks

```bash
# Run all 5 database benchmarks (~30-60 min on laptop)
python src/runner.py \
    --corpus-embeddings data/corpus_embeddings.npy \
    --query-embeddings data/query_embeddings.npy \
    --qrels data/qrels.tsv \
    --output results/results.csv

# Output:
# results/results.csv with columns:
#  - database, index_time_sec, disk_mb
#  - latency_p50_ms, latency_p95_ms, latency_p99_ms
#  - recall_at_10, mrr_at_10, num_queries
```

### 6. Generate Plots

```bash
# Create visualizations
python src/plot.py \
    --input results/results.csv \
    --output results/

# Generated plots:
# ✓ results/pareto_frontier.png          (recall vs latency)
# ✓ results/latency_distribution.png    (p50/p95/p99)
# ✓ results/disk_size.png               (index size comparison)
# ✓ results/index_time.png              (construction time)
# ✓ results/recall_vs_indexing.png      (quality vs speed trade-off)
```

### 7. View Results

```bash
# Open plots (platform-specific)
# macOS:  open results/pareto_frontier.png
# Linux:  xdg-open results/pareto_frontier.png
# Windows: start results\pareto_frontier.png

# Or view the CSV directly
cat results/results.csv
```

### 8. Cleanup

```bash
# Stop Docker services
docker compose down

# Remove temporary files (optional)
rm -rf data/corpus_embeddings.npy data/query_embeddings.npy
rm -rf /tmp/faiss_*.index /tmp/chroma_data
```

## 📁 Project Structure

```
homework-vector-db-benchmark/
├── src/
│   ├── benchmarks/           # Vector DB implementations
│   │   ├── base.py          # Abstract interface
│   │   ├── faiss_flat.py    # FAISS Flat (100% recall)
│   │   ├── faiss_hnsw.py    # FAISS HNSW (approximate)
│   │   ├── qdrant_db.py     # Qdrant client
│   │   ├── pgvector_db.py   # PostgreSQL + pgvector
│   │   └── chroma_db.py     # Chroma embedded DB
│   ├── load_data.py         # Download BeIR/quora
│   ├── embed.py             # Generate embeddings
│   ├── metrics.py           # Recall, MRR, NDCG
│   ├── runner.py            # Benchmark orchestrator
│   └── plot.py              # Visualize results
├── data/
│   ├── corpus.jsonl         # 523K documents
│   ├── queries.jsonl        # 10K queries
│   ├── qrels.tsv            # Ground truth
│   ├── corpus_embeddings.npy      # 384-dim vectors
│   └── query_embeddings.npy       # Query vectors
├── results/
│   ├── results.csv          # Benchmark results
│   ├── pareto_frontier.png  # Key visualization
│   └── *.png                # Other plots
├── docker-compose.yml       # Qdrant + PostgreSQL
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 📊 Metrics Explained

### Recall@10
- Proportion of relevant documents in top-10 results
- 1.0 = perfect (all relevant docs in top 10)
- Formula: `|retrieved[:10] ∩ relevant| / min(10, |relevant|)`

### MRR@10 (Mean Reciprocal Rank)
- Position of first relevant result (1/rank)
- 1.0 = first result is relevant
- 0.5 = second result is relevant
- Formula: `1 / position_of_first_relevant`

### Latency (p50, p95, p99)
- Query latency percentiles in milliseconds
- p50 = median latency
- p95 = 95th percentile (affects user experience)
- p99 = 99th percentile (tail latency)

### Index Time
- Seconds to build index on 523K vectors
- Important for real-time reindexing scenarios

### Disk Size
- Index size on disk in MB
- Relevant for storage constraints

## 🔧 Advanced Configuration

### Use OpenAI Embeddings (instead of local model)

```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-..."

# Use text-embedding-3-small (1536-dim)
python src/embed.py \
    --model openai/text-embedding-3-small \
    --corpus-path data/corpus.jsonl \
    --corpus-embeddings data/corpus_embeddings_openai.npy
```

### Tune FAISS HNSW Parameters

```python
# In runner.py, modify:
db = FAISSHNSw(ef_construction=400)  # Higher = better quality, slower
```

### Use Different Embedding Model

```bash
python src/embed.py \
    --model "all-MiniLM-L6-v2" \
    --corpus-embeddings data/corpus_embeddings_minilm.npy
```

### Run Subset of Databases

```python
# In runner.py, modify databases dict:
databases = {
    "FAISS Flat": FAISSFlat(),
    "Qdrant": QdrantDB(),
    # Remove others
}
```

## 🐛 Troubleshooting

### Docker services won't start
```bash
# Check Docker is running
docker --version

# Check ports aren't in use
lsof -i :6333  # Qdrant
lsof -i :5432  # PostgreSQL

# View logs
docker compose logs qdrant
docker compose logs postgres
```

### Out of memory errors
- Reduce batch size: `--batch-size 16`
- Use subset of data
- Disable FAISS Flat (most memory-intensive)

### Embedding generation is slow
- Reduce batch size for more stable memory
- Use GPU if available (install `torch[cuda]`)
- Or use OpenAI API (costs ~$0.10)

### Connection refused errors
```bash
# Make sure Docker services are running
docker compose ps

# If not running:
docker compose up -d

# Check services are healthy
curl http://localhost:6333/health
psql -h localhost -U bench -d bench -c "SELECT 1"
```

## 📈 Expected Results

On a modern laptop with CPU:

| Database | Recall | p50 | p95 | p99 | Index Time | Disk |
|----------|--------|-----|-----|-----|-----------|------|
| FAISS Flat | 1.000 | ~5.0ms | ~6.0ms | ~7.0ms | ~10s | ~2GB |
| FAISS HNSW | 0.985 | ~0.5ms | ~0.7ms | ~1.0ms | ~15s | ~500MB |
| Qdrant | 0.980 | ~2.0ms | ~3.0ms | ~4.0ms | ~20s | - |
| pgvector | 0.978 | ~1.5ms | ~2.5ms | ~3.5ms | ~30s | ~600MB |
| Chroma | 0.975 | ~1.2ms | ~1.8ms | ~2.5ms | ~25s | ~400MB |

*Actual results depend on hardware and embedding model*

## 📚 Key Learning Points

1. **FAISS Flat** guarantees 100% recall but is slow (exhaustive search)
2. **HNSW** provides 95-99% recall with 10-100x speedup via approximate search
3. **Latency scales** with index construction complexity (Flat fastest, pgvector slowest)
4. **Disk size varies** widely (FAISS HNSW ~240MB, pgvector ~600MB for same data)
5. **Network overhead** dominates for client-server systems (Qdrant, pgvector)
6. **Embedding quality** more important than DB choice (same embeddings across all)

## 🎯 Use Cases & Recommendations

- **Real-time prod** (sub-10ms): FAISS HNSW or Chroma
- **SQL queries + vectors**: pgvector
- **Microservice architecture**: Qdrant
- **Embedded/offline**: Chroma
- **Maximum accuracy**: FAISS Flat (or use larger k for approximate)

## 📖 Further Reading

- [HNSW Paper](https://arxiv.org/abs/1802.02413)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Chroma Docs](https://docs.trychroma.com/)
- [BeIR Benchmark](https://github.com/beir-cellar/beir)

## 📝 License

This is homework for AI Engineering course. Feel free to use and modify for educational purposes.

---

**Questions?** Check `IMPLEMENTATION_GUIDE.md` for technical details on each database implementation.
