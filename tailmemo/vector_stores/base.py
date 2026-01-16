from abc import ABC, abstractmethod
from enum import Enum


class DistanceMetric(str, Enum):
    L2 = "<->"
    COSINE = "<=>"
    INNER_PRODUCT = "<#>"


class VectorStoreBase(ABC):
    @abstractmethod
    def list_cols(self):
        """List all collections."""
        pass

    @abstractmethod
    def create_col(self):
        """Create a new collection."""
        pass

    @abstractmethod
    def search(self, query, vectors, limit=5, filters=None, distance_metric=DistanceMetric.COSINE):
        """Search for similar vectors."""
        pass

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None):
        """Insert vectors into a collection."""
        pass
