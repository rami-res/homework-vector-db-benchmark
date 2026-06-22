import numpy as np
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
import shutil
import os
from .base import VectorDB


class ChromaDB(VectorDB):
    """Chroma embedded vector database with persistent storage."""

    def __init__(self, persist_path: str = "./chroma_data"):
        """
        Initialize Chroma client with persistent storage.
        persist_path: directory to store Chroma data
        """
        self.persist_path = persist_path
        self.client = None
        self.collection = None
        self.vector_dim = None
        self.id_count = 0  # Chroma requires numeric IDs internally

    def _get_client(self):
        """Get or create Chroma client."""
        if self.client is None:
            # Create settings for persistent storage
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_path,
                anonymized_telemetry=False,
            )
            # Initialize client with persistent settings
            self.client = chromadb.Client(settings)
        return self.client

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build Chroma collection from vectors.
        vectors: shape (N, dim), float32
        ids: list of string IDs parallel to vectors
        """
        assert vectors.dtype == np.float32, "vectors must be float32"
        assert len(vectors) == len(ids), "vectors and ids lengths must match"

        self.vector_dim = vectors.shape[1]

        client = self._get_client()

        # Delete collection if exists
        try:
            client.delete_collection(name="benchmark")
        except Exception:
            pass

        # Create collection
        self.collection = client.create_collection(
            name="benchmark",
            metadata={"hnsw:space": "cosine"},
        )

        # Add vectors in batches
        batch_size = 1000
        for i in range(0, len(vectors), batch_size):
            batch_ids = [str(j) for j in range(i, min(i + batch_size, len(vectors)))]
            batch_vectors = vectors[i : i + batch_size].tolist()
            batch_doc_ids = ids[i : i + batch_size]

            # Add to collection with metadata containing original doc_id
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_vectors,
                metadatas=[{"doc_id": doc_id} for doc_id in batch_doc_ids],
            )

            self.id_count = i + batch_size

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbors in Chroma.
        query_vec: shape (dim,), 1D array
        Returns: [(doc_id, score), ...] of length top_k
        """
        assert self.collection is not None, "index() must be called first"
        assert query_vec.ndim == 1, "query_vec must be 1D"

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
        )

        # Extract results from the response
        # results contains: ids, distances, embeddings, metadatas
        output = []

        if results["ids"] and len(results["ids"]) > 0:
            ids_list = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []

            for idx, (chroma_id, distance, metadata) in enumerate(
                zip(ids_list, distances, metadatas)
            ):
                # Chroma returns distance, convert to similarity
                # For cosine: distance = 1 - similarity, so similarity = 1 - distance
                doc_id = metadata.get("doc_id", chroma_id)
                similarity = 1.0 - float(distance)
                output.append((doc_id, similarity))

        return output[:top_k]

    def disk_size_mb(self) -> float:
        """
        Return total size of Chroma persistent directory in MB.
        """
        if os.path.exists(self.persist_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.persist_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        return 0.0

    def cleanup(self) -> None:
        """Delete Chroma persistent directory."""
        if os.path.exists(self.persist_path):
            shutil.rmtree(self.persist_path)
        self.client = None
        self.collection = None
