import numpy as np
from typing import List, Tuple
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from .base import VectorDB


class QdrantDB(VectorDB):
    """Qdrant vector database client (requires Docker container running)."""

    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "benchmark"):
        """
        Initialize Qdrant client.
        url: Qdrant server URL
        collection_name: name of the collection to use
        """
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_dim = None

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build Qdrant collection from vectors.
        vectors: shape (N, dim), float32
        ids: list of string IDs parallel to vectors
        """
        assert vectors.dtype == np.float32, "vectors must be float32"
        assert len(vectors) == len(ids), "vectors and ids lengths must match"

        self.vector_dim = vectors.shape[1]

        # Delete collection if exists
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # Collection doesn't exist yet

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=Distance.COSINE,
            ),
        )

        # Upload via generator to avoid building large lists in memory
        def points_generator():
            for i, (doc_id, vector) in enumerate(zip(ids, vectors)):
                yield PointStruct(
                    id=i,  # Use sequential int ID; store original in payload
                    vector=vector.tolist(),
                    payload={"doc_id": doc_id},
                )

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points_generator(),
            batch_size=256,
        )

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbors in Qdrant.
        query_vec: shape (dim,), 1D array
        Returns: [(doc_id, score), ...] of length top_k
        """
        assert self.vector_dim is not None, "index() must be called first"
        assert query_vec.ndim == 1, "query_vec must be 1D"

        # Search in Qdrant
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec.tolist(),
            limit=top_k,
        )

        # Convert to expected format
        output = []
        for scored_point in response.points:
            doc_id = scored_point.payload["doc_id"]
            score = float(scored_point.score)
            output.append((doc_id, score))

        return output

    def disk_size_mb(self) -> float:
        """Estimate collection size in MB from vectors_count * dim * 4 bytes (float32)."""
        info = self.client.get_collection(self.collection_name)
        vectors_count = info.points_count or 0
        dim = self.vector_dim or 0
        return (vectors_count * dim * 4) / (1024 * 1024)

    def cleanup(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
