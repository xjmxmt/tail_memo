import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from tailmemo.vector_stores.configs import VectorStoreConfig
from tailmemo.llms.configs import LlmConfig
from tailmemo.embeddings.configs import EmbedderConfig
from tailmemo.graphs.configs import GraphStoreConfig
from tailmemo.rerankers.configs import RerankerConfig

# Set up the directory path
home_dir = os.path.expanduser("~")
tailmemo_dir = os.environ.get("TAILMEMO_DIR") or os.path.join(home_dir, ".tailmemo")


class MemoryItem(BaseModel):
    id: str = Field(..., description="The unique identifier for the text file")
    memory: str = Field(
        ..., description="The memory deduced from the text file"
    )  # TODO After prompt changes from platform, update this
    hash: Optional[str] = Field(None, description="The hash of the memory")
    # The metadata value can be anything and not just string. Fix it
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the text file")
    score: Optional[float] = Field(None, description="The score associated with the text file")
    created_at: Optional[str] = Field(None, description="The timestamp when the memory was created")
    updated_at: Optional[str] = Field(None, description="The timestamp when the memory was updated")


class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    history_db_path: str = Field(
        description="Path to the history database",
        default=os.path.join(tailmemo_dir, "history.db"),
    )
    graph_store: GraphStoreConfig = Field(
        description="Configuration for the graph",
        default_factory=GraphStoreConfig,
    )
    reranker: Optional[RerankerConfig] = Field(
        description="Configuration for the reranker",
        default=None,
    )
    version: str = Field(
        description="The version of the API",
        default="v1.1",
    )
    custom_fact_extraction_prompt: Optional[str] = Field(
        description="Custom prompt for the fact extraction",
        default=None,
    )
    custom_update_memory_prompt: Optional[str] = Field(
        description="Custom prompt for the update memory",
        default=None,
    )
    enable_perf_logging: bool = Field(
        description="Performance logging, default to False",
        default=False,
    )
