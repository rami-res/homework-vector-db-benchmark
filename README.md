# homework-vector-db-benchmark

# 1. Setup

```sh
git clone git@github.com:rami-res/homework-vector-db-benchmark.git
cd homework-vector-db-benchmark
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

# 2. Підняти Qdrant + Postgres з pgvector у Docker

```sh
docker compose up -d
docker compose ps    # перевір що hw_qdrant і hw_pgvector запущені
```

# 3. Завантажити датасет BeIR/quora (~5 хв, один раз)

```sh
python src/load_data.py
# → data/corpus.jsonl    (~523K документів)
# → data/queries.jsonl   (~10K запитів)
# → data/qrels.tsv       (golden labels)
```

# 4. Згенерувати embeddings (~10-30 хв залежно від CPU, один раз)

```sh
python src/embed.py \
    --model BAAI/bge-small-en-v1.5 \
    --input data/corpus.jsonl \
    --output data/embeddings.npy
```

# 5. Запустити бенчмарк всіх 5 БД (~30-60 хв на ноуті)

```sh
python src/runner.py --output results/results.csv
```

# 6. Згенерувати графіки зі скріншотами

```sh
python src/plot.py --input results/results.csv --output results/
```

# 7. Переглянути результати (cross-platform)

```sh
# macOS:  
open results/pareto_frontier.png

# Linux:  

xdg-open results/pareto_frontier.png

# Win:    
start results\pareto_frontier.png
```

# 8. Зупинити Docker після бенчмарка

```sh
docker compose down
```
