from typing import List, Literal, Optional

from tailmemo.embeddings.base import EmbeddingBase


class MockEmbeddings(EmbeddingBase):
    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Generate a mock embedding with dimension of 10.
        """
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def embed_batch(self, texts: List[str], memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[List[float]]:
        """
        Generate mock embeddings for multiple texts.
        
        Args:
            texts (List[str]): The texts to embed.
            memory_action (optional): The type of embedding to use. Defaults to None.
        Returns:
            List[list]: List of mock embedding vectors.
        """
        return [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] for _ in texts]
