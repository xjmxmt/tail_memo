<p align="center">
  <img src="visualization/frontend/public/tailai_logo.png" alt="TailAI Logo" width="80">
  <p align="center">
    AI 长期记忆管理系统 - 小说、戏剧、电影剧本创作 | AI Long-term Memory Management System - For novel, play, screen writing.
  </p>
</p>

<p align="center">
  <a href="#overview">English</a> | <a href="#概述">中文</a>
</p>


## Overview

**Tailmemo** is a powerful AI memory management system designed to provide long-term memory capabilities for AI applications in novel, play, screen writing. It automatically extracts facts from conversations or text snippets, stores them in vector and graph databases, and enables intelligent retrieval with reasoning support.

This project is inspired by and references the architecture of [mem0](https://github.com/mem0ai/mem0).

## ✨ Features

- **Intelligent Fact Extraction**: Automatically extract structured facts from text using LLMs
- **Hybrid Storage**: Combine vector storage (pgvector) and graph database (Neo4j) for comprehensive memory
- **Knowledge Graph**: Build entity relationships and enable graph-based reasoning queries
- **Semantic Search**: Find relevant memories using embedding-based similarity search
- **Async Support**: Full async/await support for high-performance applications
- **Flexible Integration**: Support for multiple LLM providers (OpenAI, DeepSeek, Qwen, etc.)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Tailmemo                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Memory    │──> │     LLM     │───>│   Fact Extraction   │  │
│  │   (Core)    │    │  Provider   │    │      & Update       │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│        │                                         │              │
│        ▼                                         ▼              │
│  ┌─────────────┐                       ┌─────────────────────┐  │
│  │  Embedder   │                       │    Graph Store      │  │
│  │ (Embedding) │                       │      (Neo4j)        │  │
│  └─────────────┘                       └─────────────────────┘  │
│        │                                         │              │
│        ▼                                         ▼              │
│  ┌─────────────┐                       ┌─────────────────────┐  │
│  │Vector Store │                       │ Entity & Relation   │  │
│  │ (pgvector)  │                       │    Extraction       │  │
│  └─────────────┘                       └─────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **Memory** | Core module handling memory add/search/update operations |
| **Vector Store** | PostgreSQL with pgvector extension for semantic similarity search |
| **Graph Store** | Neo4j graph database for entity relationships and reasoning |
| **LLM** | Language model integration via LangChain (supports various providers) |
| **Embedder** | Text embedding generation (OpenAI, DashScope, etc.) |
| **Reranker** | Optional result reranking for improved relevance |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- Neo4j database

> 💡 Or you can quickly start both PostgreSQL (with pgvector) and Neo4j using Docker:
> ```bash
> cd visualization
> docker-compose up -d
> ```

### Installation

```bash
# Clone the repository
git clone https://github.com/xjmxmt/tail_memo.git
cd tailmemo

# Install dependencies
pip install -r requirements.txt
```

### Configuration Example

```python
from tailmemo.memory.main import Memory

config = {
    # LLM Configuration
    "llm": {
        "provider": "langchain",
        "config": {
            "model": your_llm_instance  # LangChain ChatModel
        }
    },
    # Embedder Configuration
    "embedder": {
        "provider": "dashscope",  # or "openai"
    },
    # Vector Store Configuration
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "user": "your_user",
            "password": "your_password",
            "host": "localhost",
            "port": 5432,
            "embedding_model_dims": 1536
        }
    },
    # Graph Store Configuration
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "your_password"
        }
    }
}

# Initialize Memory
m = Memory.from_config(config)
```

### Usage

```python
import asyncio

# Add memory
result = asyncio.run(m.add(
    "The protagonist is named John, he is a detective with a keen eye for detail, he is married to Sam.",
    user_id="user_001",
    metadata={"chapter": 1}
))

# Search memory
search_results = asyncio.run(m.search(
    "What is John's profession?",
    user_id="user_001"
))

# Graph-based search with reasoning
graph_results = asyncio.run(m.graph.search_with_reasoning(
    "What relationships does John have?",
    filters={"user_id": "user_001"}
))
```

## 📁 Project Structure

```
tailmemo/
├── tailmemo/
│   ├── memory/          # Core memory management
│   ├── graphs/          # Graph memory (Neo4j)
│   ├── vector_stores/   # Vector storage (pgvector)
│   ├── llms/            # LLM providers
│   ├── embeddings/      # Embedding models
│   ├── rerankers/       # Reranking models
│   ├── configs/         # Configuration schemas
│   ├── storage/         # SQLite history storage
│   └── utils/           # Utility functions
├── tests/               # Test cases
├── evaluation/          # Evaluation benchmarks
└── visualization/       # Web UI for visualization
```

## 🙏 Acknowledgments

This project is inspired by and references the excellent work of [mem0](https://github.com/mem0ai/mem0).

---

## 概述

**Tailmemo** 是一个强大的 AI 记忆管理系统，旨在为小说、戏剧、电影剧本创作等领域的 AI 应用提供长期记忆能力。它能够自动从对话、文字片段中提取事实，将其存储在向量数据库和图数据库中，并支持带有推理能力的智能检索。

本项目的设计参考了 [mem0](https://github.com/mem0ai/mem0) 项目的架构。

## ✨ 特性

- **智能事实提取**：使用大语言模型自动从文本中提取结构化事实
- **混合存储**：结合向量存储（pgvector）和图数据库（Neo4j）实现全面的记忆管理
- **知识图谱**：构建实体关系图谱，支持基于图的推理查询
- **语义搜索**：基于嵌入向量的相似度检索相关记忆
- **异步支持**：完整的 async/await 支持，适用于高性能应用场景
- **灵活集成**：支持多种 LLM 提供商（OpenAI、DeepSeek、通义千问等）

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                           Tailmemo                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Memory    │──> │     LLM     │───>│   Fact Extraction   │  │
│  │   (Core)    │    │  Provider   │    │      & Update       │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│        │                                         │              │
│        ▼                                         ▼              │
│  ┌─────────────┐                       ┌─────────────────────┐  │
│  │  Embedder   │                       │    Graph Store      │  │
│  │ (Embedding) │                       │      (Neo4j)        │  │
│  └─────────────┘                       └─────────────────────┘  │
│        │                                         │              │
│        ▼                                         ▼              │
│  ┌─────────────┐                       ┌─────────────────────┐  │
│  │Vector Store │                       │ Entity & Relation   │  │
│  │ (pgvector)  │                       │    Extraction       │  │
│  └─────────────┘                       └─────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 描述 |
|------|------|
| **Memory** | 核心模块，处理记忆的添加、搜索、更新操作 |
| **Vector Store** | 基于 PostgreSQL + pgvector 的向量存储，支持语义相似度搜索 |
| **Graph Store** | Neo4j 图数据库，用于存储实体关系和支持推理查询 |
| **LLM** | 通过 LangChain 集成各种大语言模型（支持多种提供商） |
| **Embedder** | 文本嵌入向量生成（支持 OpenAI、DashScope 等） |
| **Reranker** | 可选的结果重排序，提升检索相关性 |

## 🚀 快速开始

### 前置要求

- Python 3.10+
- PostgreSQL（需安装 pgvector 扩展）
- Neo4j 数据库

> 💡 也可以使用 Docker 快速启动 PostgreSQL（含 pgvector）和 Neo4j：
> ```bash
> cd visualization
> docker-compose up -d
> ```

### 安装

```bash
# 克隆仓库
git clone https://github.com/xjmxmt/tail_memo.git
cd tailmemo

# 安装依赖
pip install -r requirements.txt
```

### 配置

```python
from tailmemo.memory.main import Memory

config = {
    # LLM 配置
    "llm": {
        "provider": "langchain",
        "config": {
            "model": your_llm_instance  # LangChain ChatModel 实例
        }
    },
    # 嵌入模型配置
    "embedder": {
        "provider": "dashscope",  # 或 "openai"
    },
    # 向量存储配置
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "user": "your_user",
            "password": "your_password",
            "host": "localhost",
            "port": 5432,
            "embedding_model_dims": 1536
        }
    },
    # 图存储配置
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "your_password"
        }
    }
}

# 初始化 Memory
m = Memory.from_config(config)
```

### 使用示例

```python
import asyncio

# 添加记忆
result = asyncio.run(m.add(
    "主角叫高启强，他本来是个卖鱼的，后来成为了黑帮老大，和陈书婷结婚了。",
    user_id="user_001",
    metadata={"chapter": 1}
))

# 搜索记忆
search_results = asyncio.run(m.search(
    "高启强的职业是什么？",
    user_id="user_001"
))

# 基于图的推理搜索
graph_results = asyncio.run(m.graph.search_with_reasoning(
    "高启强有哪些人际关系？",
    filters={"user_id": "user_001"}
))
```

## 📁 项目结构

```
tailmemo/
├── tailmemo/
│   ├── memory/          # 核心记忆管理
│   ├── graphs/          # 图记忆（Neo4j）
│   ├── vector_stores/   # 向量存储（pgvector）
│   ├── llms/            # LLM 提供商
│   ├── embeddings/      # 嵌入模型
│   ├── rerankers/       # 重排序模型
│   ├── configs/         # 配置模式
│   ├── storage/         # SQLite 历史存储
│   └── utils/           # 工具函数
├── tests/               # 测试用例
├── evaluation/          # 评估基准
└── visualization/       # 可视化 Web UI
```

## 🙏 致谢

本项目的设计参考了 [mem0](https://github.com/mem0ai/mem0) 项目的优秀工作。

---

## License

Apache 2.0
