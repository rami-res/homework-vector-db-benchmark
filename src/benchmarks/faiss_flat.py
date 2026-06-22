import numpy as np
import faiss
from typing import List, Tuple
import os
from .base import VectorDB


class FAISSFlat(VectorDB):
    """FAISS Flat index — exhaustive search, guaranteed 100% recall."""

    def __init__(self, output_path: str = "faiss_flat.index"):
        self.output_path = output_path
        self.index = None
        self.id_map = {}  # doc_id -> internal index

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build FAISS Flat index from vectors.
        vectors: shape (N, dim), float32
        ids: list of string IDs parallel to vectors
        """
        assert vectors.dtype == np.float32, "vectors must be float32"
        assert len(vectors) == len(ids), "vectors and ids lengths must match"

        dim = vectors.shape[1]

        # Create Flat index (L2 distance)
        self.index = faiss.IndexFlatL2(dim)

        # Add vectors
        self.index.add(vectors)

        # Store ID mapping
        self.id_map = {doc_id: idx for idx, doc_id in enumerate(ids)}

        # Save to disk
        faiss.write_index(self.index, self.output_path)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbors.
        query_vec: shape (dim,), 1D array
        Returns: [(doc_id, distance), ...] of length top_k
        """
        assert self.index is not None, "index() must be called first"
        assert query_vec.ndim == 1, "query_vec must be 1D"

        # Reshape to (1, dim)
        query_batch = query_vec.reshape(1, -1).astype(np.float32)

        # Search
        distances, indices = self.index.search(query_batch, top_k)

        # Convert back to doc_ids
        # Reverse the ID map
        idx_to_id = {v: k for k, v in self.id_map.items()}

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc_id = idx_to_id[idx]
            # Convert L2 distance to similarity (lower distance = higher similarity)
            # For consistency, use negative distance as score (higher is better)
            results.append((doc_id, -float(distance)))

        return results

    def disk_size_mb(self) -> float:
        """Return index size in MB."""
        if os.path.exists(self.output_path):
            return os.path.getsize(self.output_path) / (1024 * 1024)
        return 0.0

    def cleanup(self) -> None:
        """Remove index file."""
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
