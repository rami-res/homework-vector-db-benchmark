import numpy as np
import faiss
from typing import List, Tuple
import os
from .base import VectorDB


class FAISSHNSw(VectorDB):
    """FAISS with HNSW (Hierarchical Navigable Small World) index."""

    def __init__(self, output_path: str = "faiss_hnsw.index", ef_construction: int = 200):
        """
        Initialize HNSW index.
        ef_construction: parameter for HNSW construction (higher = more accurate but slower)
        """
        self.output_path = output_path
        self.index = None
        self.id_map = {}
        self.ef_construction = ef_construction

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build HNSW index from vectors.
        vectors: shape (N, dim), float32
        ids: list of string IDs parallel to vectors
        """
        assert vectors.dtype == np.float32, "vectors must be float32"
        assert len(vectors) == len(ids), "vectors and ids lengths must match"

        dim = vectors.shape[1]

        # Create HNSW index wrapped in IndexIDMap
        # HNSW(M=16) is a good balance
        hnsw_index = faiss.IndexHNSWFlat(dim, 16)
        hnsw_index.ef_construction = self.ef_construction
        hnsw_index.ef = min(self.ef_construction, 100)  # ef for search

        # Add vectors
        hnsw_index.add(vectors)

        # Store in class
        self.index = hnsw_index

        # Store ID mapping
        self.id_map = {doc_id: idx for idx, doc_id in enumerate(ids)}

        # Save to disk
        faiss.write_index(self.index, self.output_path)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbors using HNSW.
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
        idx_to_id = {v: k for k, v in self.id_map.items()}

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc_id = idx_to_id[idx]
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
