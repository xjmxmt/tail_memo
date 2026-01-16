import os
from typing import List, Literal, Optional

from tailmemo.configs.embeddings.base import BaseEmbedderConfig
from tailmemo.embeddings.base import EmbeddingBase

try:
    import dashscope
except ImportError:
    raise ImportError("dashscope is not installed. "
                      "Please install it using `pip install dashscope`")


class DashScopeEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-v4"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        self.api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY")

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using DashScope.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """

        return (
            dashscope.TextEmbedding.call(
                model=self.config.model,
                input=text,
                dimension=self.config.embedding_dims,
                output_type="dense"
            )["output"]["embeddings"][0]["embedding"]
        )

    def embed_batch(self, texts: List[str], memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a single batch API call using DashScope.

        Args:
            texts (List[str]): The texts to embed.
            memory_action (optional): The type of embedding to use. Defaults to None.
        Returns:
            List[list]: List of embedding vectors in the same order as input texts.
        """
        if not texts:
            return []

        # DashScope supports batch embedding with input as list
        response = dashscope.TextEmbedding.call(
            model=self.config.model,
            input=texts,
            dimension=self.config.embedding_dims,
            output_type="dense"
        )

        # Extract embeddings and return in input order
        embeddings = response["output"]["embeddings"]
        # Sort by text_index to ensure correct order
        sorted_embeddings = sorted(embeddings, key=lambda x: x.get("text_index", 0))
        return [item["embedding"] for item in sorted_embeddings]
