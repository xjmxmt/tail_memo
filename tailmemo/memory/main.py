import asyncio
import pytz
import uuid
import hashlib
import json
from loguru import logger
from datetime import datetime
from copy import deepcopy
from typing import Optional, Dict, Any
import time

from pydantic import ValidationError

from tailmemo.exceptions import ValidationError as MemoValidationError
from tailmemo.configs.base import MemoryConfig, MemoryItem
from tailmemo.configs.enums import MemoryType
from tailmemo.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from tailmemo.storage.postgresql import PostgreSQLManager
from tailmemo.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    VectorStoreFactory,
    RerankerFactory,
)
from tailmemo.memory.utils import (
    extract_json,
    get_fact_retrieval_messages,
    parse_vision_messages,
    parse_messages,
    remove_code_blocks,
)


def _safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to JSON serialization for non-serializable objects."""
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using JSON serialization: {e}")

        config_class = type(config)

        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump(mode="json")
            except Exception:
                clone_dict = {k: v for k, v in config.__dict__.items()}
        elif hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict
            clone_dict = asdict(config)
        else:
            clone_dict = {k: v for k, v in config.__dict__.items()}

        sensitive_tokens = ("auth", "credential", "password", "token", "secret", "key", "connection_class")
        for field_name in list(clone_dict.keys()):
            if any(token in field_name.lower() for token in sensitive_tokens):
                clone_dict[field_name] = None

        try:
            return config_class(**clone_dict)
        except Exception as reconstruction_error:
            logger.warning(
                f"Failed to reconstruct config: {reconstruction_error}. "
                f"Telemetry may be affected."
            )
            raise


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise MemoValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


class Memory:
    def __init__(self, config: Optional[MemoryConfig] = None):
        if config is None:
            raise ValueError("config is required. Use Memory.from_config() to create a Memory instance.")
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = PostgreSQLManager(
            host=self.config.vector_store.config.host,
            port=self.config.vector_store.config.port,
            database=self.config.vector_store.config.dbname,
            user=self.config.vector_store.config.user,
            password=self.config.vector_store.config.password
        )
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        self.enable_graph = False

        # Performance logging, default to False
        self.enable_perf_logging = self.config.enable_perf_logging

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            self.graph = GraphStoreFactory.create(provider, self.config)
            self.enable_graph = True
        else:
            self.graph = None

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    async def _run_parallel(self, messages, processed_metadata, effective_filters, infer) -> tuple:
        t0 = time.perf_counter()
        
        vector_result, graph_result = await asyncio.gather(
            self._add_to_vector_store_async(messages, processed_metadata, effective_filters, infer),
            self._add_to_graph_async(messages, processed_metadata, effective_filters),
        )
        
        if self.enable_perf_logging:
            t1 = time.perf_counter()
            logger.info(f"[memory.add 计时] 总耗时: {t1-t0:.3f}s (vector_store + graph 并行)")
        
        return vector_result, graph_result

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory (async).

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create.
                Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories.
                Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            MemoValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            GraphStoreError: If graph store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise MemoValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} "
                        f"to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise MemoValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        # run vector and graph storage in parallel
        vector_store_result, graph_result = await self._run_parallel(
            messages, processed_metadata, effective_filters, infer
        )

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    async def _search_single_fact_async(self, fact, embedding, search_filters):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.vector_store.search(
                query=fact,
                vectors=embedding,
                limit=3,
                filters=search_filters,
            )
        )

    async def _add_to_vector_store_async(self, messages, metadata, filters, infer):
        perf_times = {}
        loop = asyncio.get_event_loop()
        
        if not infer:
            # infer=False, save raw messages directly
            t0 = time.perf_counter()
            returned_memories = []
            
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                
                # 在线程池中运行 embedding 和创建操作
                msg_embeddings = await loop.run_in_executor(
                    None, self.embedding_model.embed, msg_content, "add"
                )
                mem_id = await loop.run_in_executor(
                    None, self._create_memory, msg_content, {msg_content: msg_embeddings}, per_msg_meta
                )

                returned_memories.append({
                    "id": mem_id,
                    "memory": msg_content,
                    "event": "ADD",
                    "actor_id": actor_name if actor_name else None,
                    "role": message_dict["role"],
                })
            
            perf_times["直接存储 (infer=False)"] = time.perf_counter() - t0
            
            if self.enable_perf_logging:
                logger.info(f"[VectorStore 计时] 直接存储耗时: {perf_times['直接存储 (infer=False)']:.3f}s")
            
            return returned_memories

        parsed_messages = parse_messages(messages)

        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            # is_agent_memory = self._should_use_agent_memory_extraction(messages, metadata)
            # system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)

            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        # Async LLM invoke
        t0 = time.perf_counter()
        response = await self.llm.agenerate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        perf_times["LLM 事实提取"] = time.perf_counter() - t0

        try:
            response = remove_code_blocks(response)
            if not response.strip():
                new_retrieved_facts = []
            else:
                try:
                    new_retrieved_facts = json.loads(response)["facts"]
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    new_retrieved_facts = json.loads(extracted_json)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}
        search_filters = {}
        if filters.get("user_id"):
            search_filters["user_id"] = filters["user_id"]
        if filters.get("agent_id"):
            search_filters["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            search_filters["run_id"] = filters["run_id"]

        if new_retrieved_facts:
            # Batch embedding
            t0 = time.perf_counter()
            loop = asyncio.get_event_loop()
            all_embeddings = await loop.run_in_executor(
                None, self.embedding_model.embed_batch, new_retrieved_facts
            )
            new_message_embeddings = dict(zip(new_retrieved_facts, all_embeddings))
            perf_times["Embedding 批量计算"] = time.perf_counter() - t0
            
            # Async searching
            t0 = time.perf_counter()
            search_tasks = [
                self._search_single_fact_async(fact, new_message_embeddings[fact], search_filters)
                for fact in new_retrieved_facts
            ]
            search_results = await asyncio.gather(*search_tasks)
            perf_times["向量搜索"] = time.perf_counter() - t0
            
            for result in search_results:
                for mem in result:
                    retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})

        # Remove duplicates
        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.debug(f"Total existing retrieved memories: {len(retrieved_old_memory)}")

        # If there's not any retrieved memories, skip LLM decision
        if new_retrieved_facts and not retrieved_old_memory:
            logger.info(f"No existing retrieved memories found, directly adding {len(new_retrieved_facts)} new facts")
            t0 = time.perf_counter()
            returned_memories = []
            loop = asyncio.get_event_loop()
            
            for fact in new_retrieved_facts:
                try:
                    memory_id = await loop.run_in_executor(
                        None, self._create_memory, fact, new_message_embeddings, metadata
                    )
                    returned_memories.append({
                        "id": memory_id,
                        "memory": fact,
                        "event": "ADD",
                    })
                except Exception as e:
                    logger.error(f"Error creating memory for fact: {fact}, Error: {e}")
            
            perf_times["直接 ADD (无旧记忆)"] = time.perf_counter() - t0
            
            # Print logging
            if self.enable_perf_logging:
                total_time = sum(perf_times.values())
                logger.info(f"[VectorStore 计时] 各步骤耗时:")
                for step, elapsed in perf_times.items():
                    pct = (elapsed / total_time * 100) if total_time > 0 else 0
                    bar = "█" * int(pct / 2)
                    logger.info(f"  {step:25s} {elapsed:6.3f}s ({pct:5.1f}%) {bar}")
                logger.info(f"  {'总计':25s} {total_time:6.3f}s")
            
            return returned_memories

        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        new_memories_with_actions = {}
        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )

            try:
                # Async LLM invode - ADD or NONE
                t0 = time.perf_counter()
                response: str = await self.llm.agenerate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
                perf_times["LLM 决策"] = time.perf_counter() - t0
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""

            try:
                if not response or not response.strip():
                    logger.warning("Empty response from LLM, no memories to extract")
                    new_memories_with_actions = {}
                else:
                    response = remove_code_blocks(response)
                    new_memories_with_actions = json.loads(response)
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}

        # Memory actions
        t0 = time.perf_counter()
        returned_memories = []
        loop = asyncio.get_event_loop()
        
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        logger.info("Skipping memory entry because of empty `text` field.")
                        continue

                    event_type = resp.get("event")
                    if event_type == "ADD":
                        memory_id = await loop.run_in_executor(
                            None, self._create_memory, action_text, new_message_embeddings, metadata
                        )
                        returned_memories.append(
                            {
                                "id": memory_id,
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "NONE":
                        # TODO: pass or do something
                        pass
                        # memory_id = temp_uuid_mapping.get(resp.get("id"))
                        # if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                        #     existing_memory = await loop.run_in_executor(
                        #         None, self.vector_store.get, memory_id
                        #     )
                        #     updated_metadata = deepcopy(existing_memory.payload)
                        #     if metadata.get("agent_id"):
                        #         updated_metadata["agent_id"] = metadata["agent_id"]
                        #     if metadata.get("run_id"):
                        #         updated_metadata["run_id"] = metadata["run_id"]
                        #     updated_metadata["updated_at"] = datetime.now(pytz.timezone("Asia/Shanghai")).isoformat()
                        #
                        #     await loop.run_in_executor(
                        #         None,
                        #         lambda: self.vector_store.update(
                        #             vector_id=memory_id,
                        #             vector=None,
                        #             payload=updated_metadata,
                        #         )
                        #     )
                        #     logger.info(f"Updated session IDs for memory {memory_id}")
                        # else:
                        #     logger.info("NOOP for Memory.")
                except Exception as e:
                    logger.error(f"Error processing memory action: {resp}, Error: {e}")
        except Exception as e:
            logger.error(f"Error iterating new_memories_with_actions: {e}")

        perf_times["执行操作 (ADD/NONE)"] = time.perf_counter() - t0
        
        # Print logging
        if self.enable_perf_logging:
            total_time = sum(perf_times.values())
            logger.info(f"[VectorStore 计时] 各步骤耗时:")
            for step, elapsed in perf_times.items():
                pct = (elapsed / total_time * 100) if total_time > 0 else 0
                bar = "█" * int(pct / 2)
                logger.info(f"  {step:25s} {elapsed:6.3f}s ({pct:5.1f}%) {bar}")
            logger.info(f"  {'总计':25s} {total_time:6.3f}s")

        return returned_memories

    async def _add_to_graph_async(self, messages, processed_metadata, filters):
        if not self.enable_graph:
            return []
        
        if filters.get("user_id") is None:
            filters["user_id"] = "user"

        data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])

        t0 = time.perf_counter()
        added_entities = await self.graph.add(data, filters, metadata=processed_metadata)
        
        if self.enable_perf_logging:
            logger.info(f"[Graph 计时] graph.add 耗时: {time.perf_counter()-t0:.3f}s")

        return added_entities

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def _update_memory(self, vector_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=vector_id)
        except Exception:
            logger.error(f"Error getting memory with ID {vector_id} during update.")
            raise ValueError(f"Error getting memory with ID {vector_id}. Please provide a valid 'vector_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("Asia/Shanghai")).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" not in new_metadata and "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=vector_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {vector_id=} with {data=}")

        self.db.add_history(
            vector_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return vector_id

    def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload.get("data", "")
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("Asia/Shanghai")).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id
    
    async def _search_vector_store_async(self, loop, query, effective_filters, limit, threshold):
        t0 = time.perf_counter()
        result = await loop.run_in_executor(
            None, self._search_vector_store, query, effective_filters, limit, threshold
        )
        if self.enable_perf_logging:
            logger.info(f"[search 计时] VectorStore 搜索: {time.perf_counter()-t0:.3f}s, 返回 {len(result)} 条")
        return result
    
    async def _search_graph_async(self, loop, query, effective_filters, limit):
        if not self.enable_graph:
            return None, ""
        t0 = time.perf_counter()
        result = await self.graph.search_with_reasoning(query, effective_filters, limit)
        if self.enable_perf_logging:
            events_count = len(result[0]) if result and len(result) > 0 else 0
            logger.info(f"[search 计时] Graph 搜索: {time.perf_counter()-t0:.3f}s, 返回 {events_count} 条")
        return result

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rerank: bool = False,
    ):
        """
        Searches for memories based on a query (async).
        
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Legacy filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            filters (dict, optional): Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            rerank (bool, optional): Use reranking model or not. Defaults to False.

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """
        perf_times = {}
        t_total_start = time.perf_counter()
        
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        # Apply enhanced metadata filtering if advanced operators are detected
        if filters and self._has_advanced_operators(filters):
            processed_filters = self._process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # Simple filters, merge directly
            effective_filters.update(filters)

        loop = asyncio.get_event_loop()

        # Run vector search and graph search in parallel
        t0 = time.perf_counter()
        results = await asyncio.gather(
            self._search_vector_store_async(loop, query, effective_filters, limit, threshold),
            self._search_graph_async(loop, query, effective_filters, limit),
        )
        perf_times["并行搜索 (vector + graph)"] = time.perf_counter() - t0
        
        original_memories = results[0]
        graph_result = results[1]
        
        if graph_result and len(graph_result) == 2:
            graph_entities, graph_summary = graph_result
        else:
            graph_entities, graph_summary = [], ""

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                t0 = time.perf_counter()
                reranked_memories = await loop.run_in_executor(
                    None, self.reranker.rerank, query, original_memories, limit
                )
                original_memories = reranked_memories
                perf_times["Rerank"] = time.perf_counter() - t0
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        # Print logging
        if self.enable_perf_logging:
            total_time = time.perf_counter() - t_total_start
            logger.info(f"[memory.search 计时] 各步骤耗时:")
            for step, elapsed in perf_times.items():
                pct = (elapsed / total_time * 100) if total_time > 0 else 0
                bar = "█" * int(pct / 2)
                logger.info(f"  {step:30s} {elapsed:6.3f}s ({pct:5.1f}%) {bar}")
            logger.info(f"  {'总计':30s} {total_time:6.3f}s")
            logger.info(f"  返回结果: memories={len(original_memories)}, graph_entities={len(graph_entities)}")

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities, "graph_summary": graph_summary}

        return {"results": original_memories}

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result[key] = {operator_map[operator]: value}
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))

        return processed_filters

    def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        logger.debug(memories)

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"file", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("file", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score <= threshold:
                original_memories.append(memory_item_dict)

        return original_memories


