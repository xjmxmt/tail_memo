import importlib
from typing import Dict, Optional, Union

from tailmemo.configs.embeddings.base import BaseEmbedderConfig
from tailmemo.configs.llms.base import BaseLlmConfig
from tailmemo.configs.llms.deepseek import DeepSeekConfig
from tailmemo.configs.llms.openai import OpenAIConfig
from tailmemo.configs.rerankers.base import BaseRerankerConfig
from tailmemo.configs.rerankers.cohere import CohereRerankerConfig
from tailmemo.configs.rerankers.sentence_transformer import SentenceTransformerRerankerConfig
from tailmemo.configs.rerankers.zero_entropy import ZeroEntropyRerankerConfig
from tailmemo.configs.rerankers.llm import LLMRerankerConfig
from tailmemo.configs.rerankers.huggingface import HuggingFaceRerankerConfig
from tailmemo.embeddings.mock import MockEmbeddings


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    """
    Factory for creating LLM instances with appropriate configurations.
    Supports both old-style BaseLlmConfig and new provider-specific configs.
    """

    # Provider mappings with their config classes
    provider_to_class = {
        "openai": ("tailmemo.llms.openai.OpenAILLM", OpenAIConfig),
        "deepseek": ("tailmemo.llms.deepseek.DeepSeekLLM", DeepSeekConfig),
        "langchain": ("tailmemo.llms.langchain.LangchainLLM", BaseLlmConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseLlmConfig, Dict]] = None, **kwargs):
        """
        Create an LLM instance with the appropriate configuration.

        Args:
            provider_name (str): The provider name (e.g., 'openai', 'anthropic')
            config: Configuration object or dict. If None, will create default config
            **kwargs: Additional configuration parameters

        Returns:
            Configured LLM instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")

        class_type, config_class = cls.provider_to_class[provider_name]
        llm_class = load_class(class_type)

        # Handle configuration
        if config is None:
            # Create default config with kwargs
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            # Merge dict config with kwargs
            config.update(kwargs)
            config = config_class(**config)
        elif isinstance(config, BaseLlmConfig):
            # Convert base config to provider-specific config if needed
            if config_class != BaseLlmConfig:
                # Convert to provider-specific config
                config_dict = {
                    "model": config.model,
                    "temperature": config.temperature,
                    "api_key": config.api_key,
                    "max_tokens": config.max_tokens,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "enable_vision": config.enable_vision,
                    "vision_details": config.vision_details,
                    "http_client_proxies": config.http_client,
                }
                config_dict.update(kwargs)
                config = config_class(**config_dict)
            else:
                # Use base config as-is
                pass
        else:
            # Assume it's already the correct config type
            pass

        return llm_class(config)

    @classmethod
    def register_provider(cls, name: str, class_path: str, config_class=None):
        """
        Register a new provider.

        Args:
            name (str): Provider name
            class_path (str): Full path to LLM class
            config_class: Configuration class for the provider (defaults to BaseLlmConfig)
        """
        if config_class is None:
            config_class = BaseLlmConfig
        cls.provider_to_class[name] = (class_path, config_class)

    @classmethod
    def get_supported_providers(cls) -> list:
        """
        Get list of supported providers.

        Returns:
            list: List of supported provider names
        """
        return list(cls.provider_to_class.keys())


class EmbedderFactory:
    provider_to_class = {
        "openai": "tailmemo.embeddings.openai.OpenAIEmbedding",
        "ollama": "tailmemo.embeddings.ollama.OllamaEmbedding",
        "huggingface": "tailmemo.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "tailmemo.embeddings.azure_openai.AzureOpenAIEmbedding",
        "gemini": "tailmemo.embeddings.gemini.GoogleGenAIEmbedding",
        "vertexai": "tailmemo.embeddings.vertexai.VertexAIEmbedding",
        "together": "tailmemo.embeddings.together.TogetherEmbedding",
        "lmstudio": "tailmemo.embeddings.lmstudio.LMStudioEmbedding",
        "langchain": "tailmemo.embeddings.langchain.LangchainEmbedding",
        "aws_bedrock": "tailmemo.embeddings.aws_bedrock.AWSBedrockEmbedding",
        "fastembed": "tailmemo.embeddings.fastembed.FastEmbedEmbedding",
        "dashscope": "tailmemo.embeddings.dashscope.DashScopeEmbedding"
    }

    @classmethod
    def create(cls, provider_name, config, vector_config: Optional[dict]):
        if provider_name == "upstash_vector" and vector_config and vector_config.enable_embeddings:
            return MockEmbeddings()
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "tailmemo.vector_stores.qdrant.Qdrant",
        "chroma": "tailmemo.vector_stores.chroma.ChromaDB",
        "pgvector": "tailmemo.vector_stores.pgvector.PGVector",
        "milvus": "tailmemo.vector_stores.milvus.MilvusDB",
        "upstash_vector": "tailmemo.vector_stores.upstash_vector.UpstashVector",
        "azure_ai_search": "tailmemo.vector_stores.azure_ai_search.AzureAISearch",
        "azure_mysql": "tailmemo.vector_stores.azure_mysql.AzureMySQL",
        "pinecone": "tailmemo.vector_stores.pinecone.PineconeDB",
        "mongodb": "tailmemo.vector_stores.mongodb.MongoDB",
        "redis": "tailmemo.vector_stores.redis.RedisDB",
        "valkey": "tailmemo.vector_stores.valkey.ValkeyDB",
        "databricks": "tailmemo.vector_stores.databricks.Databricks",
        "elasticsearch": "tailmemo.vector_stores.elasticsearch.ElasticsearchDB",
        "vertex_ai_vector_search": "tailmemo.vector_stores.vertex_ai_vector_search.GoogleMatchingEngine",
        "opensearch": "tailmemo.vector_stores.opensearch.OpenSearchDB",
        "supabase": "tailmemo.vector_stores.supabase.Supabase",
        "weaviate": "tailmemo.vector_stores.weaviate.Weaviate",
        "faiss": "tailmemo.vector_stores.faiss.FAISS",
        "langchain": "tailmemo.vector_stores.langchain.Langchain",
        "s3_vectors": "tailmemo.vector_stores.s3_vectors.S3Vectors",
        "baidu": "tailmemo.vector_stores.baidu.BaiduDB",
        "cassandra": "tailmemo.vector_stores.cassandra.CassandraDB",
        "neptune": "tailmemo.vector_stores.neptune_analytics.NeptuneAnalyticsVector",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance


class GraphStoreFactory:
    """
    Factory for creating MemoryGraph instances for different graph store providers.
    Usage: GraphStoreFactory.create(provider_name, config)
    """

    provider_to_class = {
        "memgraph": "tailmemo.graphs.memgraph_memory.MemoryGraph",
        "neptune": "tailmemo.graphs.neptune.neptunegraph.MemoryGraph",
        "neptunedb": "tailmemo.graphs.neptune.neptunedb.MemoryGraph",
        "kuzu": "tailmemo.graphs.kuzu_memory.MemoryGraph",
        "default": "tailmemo.graphs.graph_memory.MemoryGraph",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name, cls.provider_to_class["default"])
        try:
            GraphClass = load_class(class_type)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import MemoryGraph for provider '{provider_name}': {e}")
        return GraphClass(config)


class RerankerFactory:
    """
    Factory for creating reranker instances with appropriate configurations.
    Supports provider-specific configs following the same pattern as other factories.
    """

    # Provider mappings with their config classes
    provider_to_class = {
        "cohere": ("tailmemo.reranker.cohere_reranker.CohereReranker", CohereRerankerConfig),
        "sentence_transformer": ("tailmemo.reranker.sentence_transformer_reranker.SentenceTransformerReranker", SentenceTransformerRerankerConfig),
        "zero_entropy": ("tailmemo.reranker.zero_entropy_reranker.ZeroEntropyReranker", ZeroEntropyRerankerConfig),
        "llm_reranker": ("tailmemo.reranker.llm_reranker.LLMReranker", LLMRerankerConfig),
        "huggingface": ("tailmemo.reranker.huggingface_reranker.HuggingFaceReranker", HuggingFaceRerankerConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseRerankerConfig, Dict]] = None, **kwargs):
        """
        Create a reranker instance based on the provider and configuration.

        Args:
            provider_name: The reranker provider (e.g., 'cohere', 'sentence_transformer')
            config: Configuration object or dictionary
            **kwargs: Additional configuration parameters

        Returns:
            Reranker instance configured for the specified provider

        Raises:
            ImportError: If the provider class cannot be imported
            ValueError: If the provider is not supported
        """
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported reranker provider: {provider_name}")

        class_path, config_class = cls.provider_to_class[provider_name]

        # Handle configuration
        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config = config_class(**config, **kwargs)
        elif not isinstance(config, BaseRerankerConfig):
            raise ValueError(f"Config must be a {config_class.__name__} instance or dict")

        # Import and create the reranker class
        try:
            reranker_class = load_class(class_path)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import reranker for provider '{provider_name}': {e}")

        return reranker_class(config)
