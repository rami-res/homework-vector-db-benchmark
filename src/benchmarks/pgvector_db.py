import numpy as np
from typing import List, Tuple
import psycopg
from psycopg import sql
from .base import VectorDB


class PgvectorDB(VectorDB):
    """PostgreSQL with pgvector extension for HNSW indexing."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "bench",
        password: str = "bench",
        database: str = "bench",
        table_name: str = "vectors_benchmark",
    ):
        """
        Initialize PostgreSQL connection.
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table_name = table_name
        self.conn = None
        self.vector_dim = None

    def _get_connection(self):
        """Get or create database connection."""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.database,
            )
        return self.conn

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build pgvector table from vectors.
        vectors: shape (N, dim), float32
        ids: list of string IDs parallel to vectors
        """
        assert vectors.dtype == np.float32, "vectors must be float32"
        assert len(vectors) == len(ids), "vectors and ids lengths must match"

        self.vector_dim = vectors.shape[1]
        conn = self._get_connection()

        # Enable pgvector extension
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Drop table if exists
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(self.table_name)))

        # Create table (dimension must be a literal in DDL, not a parameter)
        conn.execute(
            sql.SQL("CREATE TABLE {} (id TEXT PRIMARY KEY, embedding vector({}))").format(
                sql.Identifier(self.table_name),
                sql.SQL(str(self.vector_dim)),
            )
        )

        # Insert vectors in batches
        batch_size = 1000
        for i in range(0, len(vectors), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]

            # Prepare data: convert numpy arrays to lists
            data = [
                (doc_id, vector.tolist())
                for doc_id, vector in zip(batch_ids, batch_vectors)
            ]

            with conn.cursor() as cur:
                cur.executemany(
                    sql.SQL("INSERT INTO {} (id, embedding) VALUES (%s, %s)").format(
                        sql.Identifier(self.table_name)
                    ),
                    data,
                )

        # Create HNSW index for faster search
        conn.execute(
            sql.SQL("CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops)").format(
                sql.Identifier(self.table_name)
            )
        )

        conn.commit()

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbors using pgvector.
        query_vec: shape (dim,), 1D array
        Returns: [(doc_id, score), ...] of length top_k
        """
        assert self.vector_dim is not None, "index() must be called first"
        assert query_vec.ndim == 1, "query_vec must be 1D"

        conn = self._get_connection()

        # Convert query to string format for pgvector
        query_str = "[" + ",".join(str(x) for x in query_vec) + "]"

        # Search using cosine distance (1 - cosine_similarity)
        results = conn.execute(
            sql.SQL(
                "SELECT id, (1 - (embedding <=> %s)) as similarity FROM {} ORDER BY embedding <=> %s LIMIT %s"
            ).format(sql.Identifier(self.table_name)),
            (query_str, query_str, top_k),
        ).fetchall()

        # Convert to expected format
        output = [(doc_id, float(score)) for doc_id, score in results]

        return output

    def disk_size_mb(self) -> float:
        """
        Return table size in MB.
        """
        conn = self._get_connection()

        result = conn.execute(
            sql.SQL(
                "SELECT pg_total_relation_size({}) / (1024.0 * 1024.0) as size_mb"
            ).format(sql.Literal(self.table_name))
        ).fetchone()

        if result:
            return float(result[0])

        return 0.0

    def cleanup(self) -> None:
        """Drop the table and close connection."""
        try:
            conn = self._get_connection()
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(self.table_name)))
            conn.commit()
        except Exception:
            pass

        if self.conn is not None and not self.conn.closed:
            self.conn.close()
