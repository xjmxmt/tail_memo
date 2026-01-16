from typing import List, Literal, Optional

from tailmemo.configs.embeddings.base import BaseEmbedderConfig
from tailmemo.embeddings.base import EmbeddingBase

try:
    from langchain.embeddings.base import Embeddings
except ImportError:
    raise ImportError("langchain is not installed. Please install it using `pip install langchain`")


class LangchainEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        if self.config.model is None:
            raise ValueError("`model` parameter is required")

        if not isinstance(self.config.model, Embeddings):
            raise ValueError("`model` must be an instance of Embeddings")

        self.langchain_model = self.config.model

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Langchain.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """

        return self.langchain_model.embed_query(text)

    def embed_batch(self, texts: List[str], memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[List[float]]:
        """
        Get embeddings for multiple texts using Langchain's embed_documents method.

        Args:
            texts (List[str]): The texts to embed.
            memory_action (optional): The type of embedding to use. Defaults to None.
        Returns:
            List[list]: List of embedding vectors in the same order as input texts.
        """
        if not texts:
            return []

        # Langchain's embed_documents is designed for batch embedding
        return self.langchain_model.embed_documents(texts)
