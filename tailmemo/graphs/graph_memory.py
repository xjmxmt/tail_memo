import time
import asyncio
import warnings
import concurrent.futures
from loguru import logger

from tailmemo.memory.utils import format_entities, sanitize_relationship_for_cypher

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from tailmemo.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
    CAUSAL_REASONING_DECISION_TOOL,
    CAUSAL_REASONING_DECISION_STRUCT_TOOL,
)
from tailmemo.configs.prompts import (
    NODE_EXTRACTION_PROMPTS, 
    NODE_EXTRACTION_PROMPTS_FOR_SEARCH,
    EXTRACT_RELATIONS_PROMPT, 
    get_delete_messages,
    get_causal_decision_messages,
    get_causal_analysis_messages,
)
from tailmemo.utils.factory import EmbedderFactory, LlmFactory


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = Neo4jGraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
            self.config.graph_store.config.database,
            refresh_schema=False,
            driver_config={"notifications_min_severity": "OFF"},
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )
        self.node_label = ":`__Entity__`" if self.config.graph_store.config.base_label else ""

        if self.config.graph_store.config.base_label:
            self._create_indexes()

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider

        # Get LLM config with proper null checks
        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        # Use threshold from graph_store config, default to 0.9
        self.threshold = self.config.graph_store.threshold if hasattr(self.config.graph_store, 'threshold') else 0.9
        
        # Performance logging, default to False
        self.enable_perf_logging = self.config.enable_perf_logging
        
        # Enable node embedding or not, off by default (disabling this can significantly improve performance)
        self.enable_node_embeddings = False

    def _create_indexes(self):
        """Automatically create Neo4j indexes"""
        indexes_to_create = [
            # Index name, Cypher statement, description
            (
                "entity_user_id",
                f"CREATE INDEX entity_user_id IF NOT EXISTS FOR (n {self.node_label}) ON (n.user_id)",
                "user_id column index"
            ),
            (
                "entity_name",
                f"CREATE INDEX entity_name IF NOT EXISTS FOR (n {self.node_label}) ON (n.name)",
                "name column index"
            ),
            (
                "entity_name_user",
                f"CREATE INDEX entity_name_user IF NOT EXISTS FOR (n {self.node_label}) ON (n.name, n.user_id)",
                "name + user_id composite index"
            ),
        ]
        
        for index_name, cypher, description in indexes_to_create:
            try:
                self.graph.query(cypher)
                logger.debug(f"索引已创建/确认: {index_name} ({description})")
            except Exception as e:
                logger.debug(f"索引 {index_name} 创建跳过: {e}")
        
        # 尝试创建向量索引（Neo4j 5.11+ 支持）
        try:
            embedding_dims = getattr(self.config.vector_store.config, 'embedding_model_dims', 1536)
            vector_index_cypher = f"""
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (n {self.node_label})
                ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {embedding_dims},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """
            self.graph.query(vector_index_cypher)
            logger.info(f"向量索引已创建 (dims={embedding_dims})")
        except Exception as e:
            logger.debug(f"向量索引创建跳过 (可能版本不支持): {e}")
        
        # 验证已创建的索引
        try:
            result = self.graph.query("SHOW INDEXES YIELD name, type, state")
            active_indexes = [r['name'] for r in result if r.get('state') == 'ONLINE']
            logger.info(f"Neo4j 活跃索引: {active_indexes}")
        except Exception:
            pass

    async def add(self, data, filters, metadata=None):
        """
        Adds data to the graph (async).

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
            metadata (dict, optional): Temporal metadata for the data.
                - chapter (int): Chapter number where this event occurs
                - sequence (int): Global sequence number for ordering events
        """
        metadata = metadata or {}
        loop = asyncio.get_event_loop()

        # Async LLM invoke
        t0 = time.perf_counter()
        to_be_added = await self._establish_nodes_relations_from_data(data, filters)
        t1 = time.perf_counter()
        if self.enable_perf_logging:
            logger.info(f"[Graph.add 计时] _establish_nodes_relations_from_data (LLM): {t1-t0:.3f}s")

        # Neo4j
        t2 = time.perf_counter()
        added_entities = await loop.run_in_executor(
            None, self._add_entities, to_be_added, filters, metadata
        )
        t3 = time.perf_counter()
        if self.enable_perf_logging:
            logger.info(f"[Graph.add 计时] _add_entities: {t3-t2:.3f}s")

        logger.debug("added_entities: {added_entities}")
        return {"added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph file.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base file store.
                - "entities": List of related graph file based on the query.
        """
        warnings.warn(
            "search() if deprecated, please use search_with_reasoning()",
            DeprecationWarning,
            stacklevel=2
        )
        entity_info_map = self._retrieve_nodes_from_data(query, filters, search=True)
        search_output = self._search_graph_db(node_list=list(entity_info_map.keys()), filters=filters, entity_info_map=entity_info_map)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    async def search_with_reasoning(self, query, filters, limit=100, max_depth=3):
        """
        Search with LLM-driven causal reasoning for long-context event understanding (async).
        
        This method uses a multi-round LLM-driven approach to:
        1. Perform initial entity extraction and search
        2. Iteratively explore the graph based on LLM decisions
        3. Analyze causal relationships in retrieved events
        4. Return results sorted by reversed temporal order
        
        Args:
            query (str): The user's question about events/causality
            filters (dict): Filter conditions (user_id, agent_id, run_id)
            limit (int): Maximum results per search round
            max_depth (int): Maximum exploration depth to prevent infinite loops
            
        Returns:
            tuple: (all_events, summary)
                - all_events: List of events sorted by temporal order
                - summary: Summary of the causal relationships
        """
        loop = asyncio.get_event_loop()
        perf_times = {}
        t_total_start = time.perf_counter()
        
        # 1: Extract initial entities from query, async LLM invoke
        t0 = time.perf_counter()
        entity_info_map = await self._retrieve_nodes_from_data_async(query, filters, search=True)
        perf_times["LLM 实体提取"] = time.perf_counter() - t0
        
        if not entity_info_map:
            if self.enable_perf_logging:
                logger.info(f"[search_with_reasoning 计时] LLM 实体提取: {perf_times['LLM 实体提取']:.3f}s, 未提取到实体")
            return [], "未能从查询中提取到实体"
        
        if self.enable_perf_logging:
            logger.info(f"[search_with_reasoning 计时] LLM 实体提取: {perf_times['LLM 实体提取']:.3f}s, 提取到 {len(entity_info_map)} 个实体: {list(entity_info_map.keys())}")
        
        # 2: Multi-round exploration with LLM decision
        all_events = []
        summary = ""
        visited_entities = set()  # Store visited entity names (strings)
        visited_events = set()    # Store visited event tuples (source, relationship, destination)
        current_entities = list(entity_info_map.keys())
        current_depth = 0
        direction = "both"
        
        total_search_time = 0.0
        total_decision_time = 0.0
        
        while current_depth < max_depth:
            # Perform search with neighbors
            t0 = time.perf_counter()
            search_results = await loop.run_in_executor(
                None,
                lambda: self._search_with_neighbors(
                    node_list=current_entities,
                    filters=filters,
                    limit=limit,
                    entity_info_map=entity_info_map,
                    direction=direction
                )
            )
            search_elapsed = time.perf_counter() - t0
            total_search_time += search_elapsed
            
            if self.enable_perf_logging:
                logger.info(f"[search_with_reasoning 计时] 第 {current_depth} 轮 Neo4j 搜索: {search_elapsed:.3f}s, 返回 {len(search_results)} 条")
            
            if not search_results:
                break
            
            # Add new events
            for event in search_results:
                event_key = (event.get("source"), event.get("relationship"), event.get("destination"))
                if event_key not in visited_events:
                    visited_events.add(event_key)
                    all_events.append(event)

            # Mark visited entities (for filtering next round)
            for event in search_results:
                visited_entities.add(event.get("source"))
                visited_entities.add(event.get("destination"))
            
            # Format retrieved info for LLM
            retrieved_info = self._format_events_for_llm(search_results)
            neighbor_summary = self._format_neighbor_summary(search_results)
            
            # Get LLM decision on whether to continue, async LLM invoke
            t0 = time.perf_counter()
            decision = await self._get_exploration_decision_async(
                query=query,
                retrieved_info=retrieved_info,
                neighbor_summary=neighbor_summary,
                current_depth=current_depth,
                max_depth=max_depth,
                visited_entities=visited_entities,
                visited_events=visited_events
            )
            decision_elapsed = time.perf_counter() - t0
            total_decision_time += decision_elapsed
            
            if self.enable_perf_logging:
                logger.info(f"[search_with_reasoning 计时] 第 {current_depth} 轮 LLM 决策: {decision_elapsed:.3f}s, action={decision.get('action')}")
            
            if decision.get("action") == "complete":
                summary = decision.get('answer')
                logger.info(f"LLM decided to complete at depth {current_depth}: reasoning: {decision.get('reasoning')},"
                            f" answer: {decision.get('answer')}")
                break
            
            # Update for next round
            next_entities = decision.get("next_entities", [])

            # Note:
            # Hardcoded searching direction to "both" for now
            # direction = decision.get("direction", "both")
            
            # Filter out already visited entities
            next_entities = [e for e in next_entities if e not in current_entities]
            current_entities = next_entities
            current_depth += 1
            
            logger.debug(f"Continuing exploration at depth {current_depth}, next entities: {current_entities}")
        
        # 3: Sort events by reversed temporal order
        all_events.sort(key=lambda x: (
            x.get("chapter") or 0,
            x.get("sequence") or 0
        ), reverse=True)

        # 打印计时汇总
        if self.enable_perf_logging:
            perf_times["Neo4j 搜索 (累计)"] = total_search_time
            perf_times["LLM 决策 (累计)"] = total_decision_time
            total_time = time.perf_counter() - t_total_start
            
            logger.info(f"[search_with_reasoning 计时] 汇总:")
            for step, elapsed in perf_times.items():
                pct = (elapsed / total_time * 100) if total_time > 0 else 0
                bar = "█" * int(pct / 2)
                logger.info(f"  {step:25s} {elapsed:6.3f}s ({pct:5.1f}%) {bar}")
            logger.info(f"  {'总计':25s} {total_time:6.3f}s")
            logger.info(f"  探索轮数: {current_depth + 1}, 返回事件: {len(all_events)}")

        return all_events, summary
    
    async def _retrieve_nodes_from_data_async(self, data, filters, search=False):
        """Extracts all the entities mentioned in the query (async version)."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        
        search_results = await self.llm.agenerate_response(
            messages=[
                {
                    "role": "system",
                    "content": NODE_EXTRACTION_PROMPTS_FOR_SEARCH if search else NODE_EXTRACTION_PROMPTS,
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_info_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_name = item["entity"]
                    entity_info_map[entity_name] = {
                        "entity_type": item.get("entity_type", "unknown"),
                        "description": item.get("description", ""),
                        "aliases": [alias for alias in item.get("aliases", [])],
                    }
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        logger.debug(f"Entity info map: {entity_info_map}\n search_results={search_results}")
        return entity_info_map
    
    async def _get_exploration_decision_async(self, query, retrieved_info, neighbor_summary, current_depth, max_depth,
                                              visited_entities, visited_events):
        """Get LLM decision on whether to continue exploration (async version)."""
        prompt = get_causal_decision_messages(
            query=query,
            retrieved_info=retrieved_info,
            neighbor_summary=neighbor_summary,
            current_depth=current_depth,
            max_depth=max_depth,
            visited_entities=visited_entities,
            visited_events=visited_events
        )
        
        _tools = [CAUSAL_REASONING_DECISION_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [CAUSAL_REASONING_DECISION_STRUCT_TOOL]
        
        try:
            response = await self.llm.agenerate_response(
                messages=[
                    {"role": "system", "content": prompt},
                ],
                tools=_tools,
            )
            
            if response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    if tool_call["name"] == "decide_next_action":
                        return tool_call.get("arguments", {})
        except Exception as e:
            logger.error(f"Error getting exploration decision: {e}")
        
        # Default: complete if error
        return {"action": "complete", "reasoning": "决策出错，默认完成"}

    def _format_events_for_llm(self, events):
        """Format events for LLM consumption."""
        if not events:
            return "无相关事件"
        
        lines = []
        for i, event in enumerate(events, 1):
            chapter = event.get("chapter", "?")
            sequence = event.get("sequence", "?")
            line = f"{i}. [{chapter}章-{sequence}序] {event['source']} --{event['relationship']}--> {event['destination']}"
            lines.append(line)
        
        return "\n".join(lines)

    def _format_neighbor_summary(self, events):
        """Format neighbor summaries for LLM."""
        lines = []
        for event in events:
            neighbors = event.get("neighbor_summary", [])
            if neighbors:
                dest = event.get("destination")
                neighbor_strs = [f"{n['name']}({n.get('relationship', '?')})" for n in neighbors[:3]]
                lines.append(f"  {dest} 的后续节点: {', '.join(neighbor_strs)}")
        
        return "\n".join(lines) if lines else "无可继续探索的邻居节点"

    def delete_all(self, filters):
        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        cypher = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100, include_deprecated=False):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
        
        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
            include_deprecated (bool): Whether to include deprecated (soft-deleted) relationships. Defaults to False.
        Returns:
            list: A list of dictionaries, each containing:
                - 'source': The source node name
                - 'relationship': The relationship type
                - 'target': The target node name
                - 'deprecated_at': (if include_deprecated=True) When the relationship was deprecated
        """
        params = {"user_id": filters["user_id"], "limit": limit}

        # Build node properties based on filters
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
            params["run_id"] = filters["run_id"]
        node_props_str = ", ".join(node_props)

        # Filter out deprecated relationships unless explicitly requested
        deprecated_filter = "" if include_deprecated else "WHERE r.deprecated_at IS NULL"
        
        query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})-[r]->(m {self.node_label} {{{node_props_str}}})
        {deprecated_filter}
        RETURN n.name AS source, type(r) AS relationship, m.name AS target,
               r.deprecated_at AS deprecated_at, r.deprecated_chapter AS deprecated_chapter
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            item = {
                "source": result["source"],
                "relationship": result["relationship"],
                "target": result["target"],
            }
            if include_deprecated and result.get("deprecated_at"):
                item["deprecated_at"] = result["deprecated_at"]
                item["deprecated_chapter"] = result.get("deprecated_chapter")
            final_results.append(item)

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters, search=False):
        """Extracts all the entities mentioned in the query.
        
        Returns:
            dict: A dictionary mapping entity names to their metadata:
                {
                    "entity_name": {
                        "entity_type": str,
                        "description": str,  # Context-aware description for enhanced embedding
                        "aliases": list[str]  # Alternative names for entity linking
                    }
                }
        """
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": NODE_EXTRACTION_PROMPTS_FOR_SEARCH if search else NODE_EXTRACTION_PROMPTS,
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_info_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_name = item["entity"]
                    entity_info_map[entity_name] = {
                        "entity_type": item.get("entity_type", "unknown"),
                        "description": item.get("description", ""),
                        "aliases": [alias for alias in item.get("aliases", [])],
                    }
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        logger.debug(f"Entity info map: {entity_info_map}\n search_results={search_results}")
        return entity_info_map

    async def _establish_nodes_relations_from_data(self, data, filters):
        """Establish relations among the extracted nodes."""

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Text: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = await self.llm.agenerate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _compute_enhanced_embedding(self, entity_name, entity_info=None):
        """Compute embedding using entity name enhanced with context description.
        
        Args:
            entity_name: The name of the entity
            entity_info: Optional dict containing 'description' and 'aliases'
            
        Returns:
            The embedding vector for the enhanced entity representation
        """
        if entity_info and entity_info.get("description"):
            # Use context-enhanced embedding: "entity_name: description"
            enhanced_text = f"{entity_name}: {entity_info['description']}"
        else:
            enhanced_text = entity_name
        return self.embedding_model.embed(enhanced_text)

    def _batch_compute_embeddings(self, entities_to_add):
        """Pre-compute embeddings for all entities in batch for better performance.
        
        Args:
            entities_to_add: List of entity dicts with source and destination info
            
        Returns:
            dict: Mapping from sanitized entity name to embedding vector
        """
        # Collect all unique entity texts that need embeddings
        texts_to_embed = []
        text_to_key = {}  # Map text to sanitized key
        
        for item in entities_to_add:
            source_info = item.get("source", {})
            destination_info = item.get("destination", {})
            
            source = source_info.get("entity", "")
            destination = destination_info.get("entity", "") if destination_info else ""
            
            # Process source
            if source and source.strip():
                sanitized_source = sanitize_relationship_for_cypher(source.lower().replace(" ", "_"))
                if sanitized_source not in text_to_key.values():
                    source_description = source_info.get("description", "")
                    if source_description:
                        enhanced_text = f"{sanitized_source}: {source_description}"
                    else:
                        enhanced_text = sanitized_source
                    texts_to_embed.append(enhanced_text)
                    text_to_key[enhanced_text] = sanitized_source
            
            # Process destination
            if destination and destination.strip():
                sanitized_dest = sanitize_relationship_for_cypher(destination.lower().replace(" ", "_"))
                if sanitized_dest not in text_to_key.values():
                    dest_description = destination_info.get("description", "") if destination_info else ""
                    if dest_description:
                        enhanced_text = f"{sanitized_dest}: {dest_description}"
                    else:
                        enhanced_text = sanitized_dest
                    texts_to_embed.append(enhanced_text)
                    text_to_key[enhanced_text] = sanitized_dest
        
        if not texts_to_embed:
            return {}
        
        # Batch compute all embeddings
        embeddings = self.embedding_model.embed_batch(texts_to_embed)
        
        # Build cache mapping sanitized name to embedding
        embedding_cache = {}
        for text, embedding in zip(texts_to_embed, embeddings):
            key = text_to_key[text]
            embedding_cache[key] = embedding
        
        logger.debug(f"Batch computed {len(embedding_cache)} embeddings")
        return embedding_cache

    def _search_graph_db(self, node_list, filters, limit=100, entity_info_map=None):
        """Search similar nodes among and their respective incoming and outgoing relations.
        
        Uses context-enhanced embeddings and supports alias matching for better entity linking.
        """
        result_relations = []
        entity_info_map = entity_info_map or {}

        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        # Collect all names to search (primary names + aliases)
        names_to_search = []
        for node in node_list:
            entity_info = entity_info_map.get(node, {})
            names_to_search.append((node, entity_info))
            # Also search for aliases
            for alias in entity_info.get("aliases", []):
                if alias != node:  # Avoid duplicates
                    names_to_search.append((alias, entity_info))

        seen_relations = set()  # Track unique relations to avoid duplicates

        for search_name, entity_info in names_to_search:
            # Use context-enhanced embedding
            n_embedding = self._compute_enhanced_embedding(search_name, entity_info)

            # Query with both vector similarity and alias matching
            cypher_query = f"""
            MATCH (n {self.node_label} {{{node_props_str}}})
            WHERE n.embedding IS NOT NULL
            WITH n, 
                 vector.similarity.cosine(n.embedding, $n_embedding) AS embedding_similarity,
                 CASE 
                     WHEN n.name = $search_name THEN 1.0
                     WHEN $search_name IN coalesce(n.aliases, []) THEN 0.95
                     ELSE 0.0 
                 END AS alias_match_score
            WITH n, 
                 CASE 
                     WHEN alias_match_score > 0 THEN alias_match_score
                     ELSE embedding_similarity
                 END AS similarity
            WHERE similarity >= $threshold
            CALL {{
                WITH n
                MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
                WHERE r.deprecated_at IS NULL
                RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id
                UNION
                WITH n  
                MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
                WHERE r.deprecated_at IS NULL
                RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id
            }}
            WITH distinct source, source_id, relationship, relation_id, destination, destination_id, similarity
            RETURN source, source_id, relationship, relation_id, destination, destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

            params = {
                "n_embedding": n_embedding,
                "search_name": search_name,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]
            if filters.get("run_id"):
                params["run_id"] = filters["run_id"]

            ans = self.graph.query(cypher_query, params=params)
            
            # Deduplicate results based on relation_id
            for item in ans:
                relation_key = item.get("relation_id")
                if relation_key and relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    result_relations.append(item)

        return result_relations

    def _search_with_neighbors(self, node_list, filters, limit=100, entity_info_map=None, direction="both"):
        """Search nodes with their relationships and neighbor summaries for causal reasoning.
        
        Returns enriched results including:
        - Relationship temporal metadata (chapter, sequence, event_time, causality)
        - Neighbor summaries (subsequent nodes for causal chain exploration)

        The search strategy is determined by self.enable_node_embeddings:
        - True: Use embedding vector similarity search + name/alias matching
        - False: Use only exact name and alias matching (faster)
        
        Args:
            node_list: List of entity names to search
            filters: Filter conditions (user_id, agent_id, run_id)
            limit: Maximum number of results
            entity_info_map: Map of entity names to their info
            direction: Search direction - "forward" (outgoing), "backward" (incoming), or "both"
        
        Returns:
            List of dicts with source, relationship, destination, temporal info, and neighbor summary
        """
        result_relations = []
        entity_info_map = entity_info_map or {}

        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        # Collect all names to search
        names_to_search = []
        for node in node_list:
            entity_info = entity_info_map.get(node, {})
            names_to_search.append((node, entity_info))
            for alias in entity_info.get("aliases", []):
                if alias != node:
                    names_to_search.append((alias, entity_info))

        seen_relations = set()

        for search_name, entity_info in names_to_search:
            # Build direction-specific match clauses (filter out deprecated relationships)
            if direction == "forward":
                match_clause = f"""
                    WITH n
                    MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
                    WHERE r.deprecated_at IS NULL
                    RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, 
                           elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id,
                           r.chapter AS chapter, r.sequence AS sequence
                """
            elif direction == "backward":
                match_clause = f"""
                    WITH n
                    MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
                    WHERE r.deprecated_at IS NULL
                    RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship,
                           elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id,
                           r.chapter AS chapter, r.sequence AS sequence
                """
            else:  # both
                match_clause = f"""
                    WITH n
                    MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
                    WHERE r.deprecated_at IS NULL
                    RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship,
                           elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id,
                           r.chapter AS chapter, r.sequence AS sequence
                    UNION
                    WITH n
                    MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
                    WHERE r.deprecated_at IS NULL
                    RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship,
                           elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id,
                           r.chapter AS chapter, r.sequence AS sequence
                """

            # Use difference searching strategies based on self. enable_node_embeddings
            if self.enable_node_embeddings:
                # Use embedding vector similarity search + name/alias matching
                n_embedding = self._compute_enhanced_embedding(search_name, entity_info)
                
                cypher_query = f"""
                MATCH (n {self.node_label} {{{node_props_str}}})
                WHERE n.embedding IS NOT NULL
                WITH n, 
                     vector.similarity.cosine(n.embedding, $n_embedding) AS embedding_similarity,
                     CASE 
                         WHEN n.name = $search_name THEN 1.0
                         WHEN $search_name IN coalesce(n.aliases, []) THEN 1.0
                         ELSE 0.0 
                     END AS alias_match_score
                WITH n, 
                     CASE 
                         WHEN alias_match_score > 0 THEN alias_match_score
                         ELSE embedding_similarity
                     END AS similarity
                WHERE similarity >= $threshold
                CALL {{
                    {match_clause}
                }}
                WITH distinct source, source_id, relationship, relation_id, destination, destination_id, 
                     chapter, sequence, similarity
                RETURN source, source_id, relationship, relation_id, destination, destination_id,
                       chapter, sequence, similarity
                ORDER BY coalesce(sequence, 0) ASC, similarity DESC
                LIMIT $limit
                """
                
                params = {
                    "n_embedding": n_embedding,
                    "search_name": search_name,
                    "threshold": self.threshold,
                    "user_id": filters["user_id"],
                    "limit": limit,
                }
            else:
                # Use only exact name and alias matching (faster)
                cypher_query = f"""
                MATCH (n {self.node_label} {{{node_props_str}}})
                WHERE n.name = $search_name 
                   OR $search_name IN coalesce(n.aliases, [])
                WITH n, 1.0 AS similarity
                CALL {{
                    {match_clause}
                }}
                WITH distinct source, source_id, relationship, relation_id, destination, destination_id, 
                     chapter, sequence, similarity
                RETURN source, source_id, relationship, relation_id, destination, destination_id,
                       chapter, sequence, similarity
                ORDER BY coalesce(sequence, 0) ASC
                LIMIT $limit
                """
                
                params = {
                    "search_name": search_name,
                    "user_id": filters["user_id"],
                    "limit": limit,
                }
            
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]
            if filters.get("run_id"):
                params["run_id"] = filters["run_id"]

            ans = self.graph.query(cypher_query, params=params)

            for item in ans:
                relation_key = item.get("relation_id")
                if relation_key and relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    result_relations.append(item)

        # Now fetch neighbor summaries for each destination node
        for relation in result_relations:
            dest_id = relation.get("destination_id")
            if dest_id:
                neighbor_summary = self._get_neighbor_summary(dest_id, filters, direction="forward", limit=3)
                relation["neighbor_summary"] = neighbor_summary

        # Sort results by sequence (reversed temporal order)
        result_relations.sort(key=lambda x: (x.get("sequence") or 0, x.get("chapter") or 0), reverse=True)

        return result_relations

    def _get_neighbor_summary(self, node_id, filters, direction="forward", limit=3):
        """Get a brief summary of neighboring nodes for a given node.
        
        Args:
            node_id: The elementId of the node
            filters: Filter conditions
            direction: "forward" for outgoing, "backward" for incoming
            limit: Maximum number of neighbors to include
            
        Returns:
            List of neighbor summaries: [{name, relationship, chapter, sequence}]
        """
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        if direction == "forward":
            cypher = f"""
            MATCH (n)
            WHERE elementId(n) = $node_id
            MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
            WHERE r.deprecated_at IS NULL
            RETURN m.name AS name, type(r) AS relationship, 
                   r.chapter AS chapter, r.sequence AS sequence
            ORDER BY coalesce(r.sequence, 0) ASC
            LIMIT $limit
            """
        else:
            cypher = f"""
            MATCH (n)
            WHERE elementId(n) = $node_id
            MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
            WHERE r.deprecated_at IS NULL
            RETURN m.name AS name, type(r) AS relationship,
                   r.chapter AS chapter, r.sequence AS sequence
            ORDER BY coalesce(r.sequence, 0) ASC
            LIMIT $limit
            """

        params = {"node_id": node_id, "user_id": filters["user_id"], "limit": limit}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        results = self.graph.query(cypher, params=params)
        return [
            {
                "name": r["name"],
                "relationship": r["relationship"],
                "chapter": r.get("chapter"),
                "sequence": r.get("sequence"),
            }
            for r in results
        ]

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Relationships to be deleted: {to_be_deleted}")
        return to_be_deleted

    def _soft_delete_entities(self, to_be_deleted, filters, metadata=None):
        """Soft delete entities by marking them as deprecated instead of removing.
        
        Sets deprecated_at timestamp and deprecated_chapter/sequence on the relationship
        to indicate when and where it became outdated.
        
        Args:
            to_be_deleted: List of relationships to deprecate
            filters: Filter conditions (user_id, agent_id, run_id)
            metadata: Temporal metadata with chapter/sequence info
        """
        metadata = metadata or {}
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        
        # Get temporal info for when this deprecation occurred
        chapter = metadata.get("chapter")
        sequence = metadata.get("sequence")
        
        results = []

        for item in to_be_deleted:
            if item.get("source", "").strip() == "":
                continue
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
                "deprecated_chapter": chapter,
                "deprecated_sequence": sequence,
            }

            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            # Build node properties for filtering
            source_props = ["name: $source_name", "user_id: $user_id"]
            dest_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_props.append("agent_id: $agent_id")
                dest_props.append("agent_id: $agent_id")
            if run_id:
                source_props.append("run_id: $run_id")
                dest_props.append("run_id: $run_id")
            source_props_str = ", ".join(source_props)
            dest_props_str = ", ".join(dest_props)

            # Soft delete: mark the relationship as deprecated instead of deleting
            cypher = f"""
            MATCH (n {self.node_label} {{{source_props_str}}})
            -[r:{relationship}]->
            (m {self.node_label} {{{dest_props_str}}})
            WHERE r.deprecated_at IS NULL
            SET r.deprecated_at = timestamp(),
                r.deprecated_chapter = $deprecated_chapter,
                r.deprecated_sequence = $deprecated_sequence
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship,
                r.deprecated_chapter AS deprecated_chapter,
                r.deprecated_sequence AS deprecated_sequence
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)
            logger.debug(f"Soft deleted relationship: {source} -[{relationship}]-> {destination}")

        return results

    def _hard_delete_entities(self, to_be_deleted, filters):
        """Hard delete entities from the graph (permanently removes relationships).
        
        Use this for cleanup operations or when soft delete is not desired.
        """
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            # Build node properties for filtering
            source_props = ["name: $source_name", "user_id: $user_id"]
            dest_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_props.append("agent_id: $agent_id")
                dest_props.append("agent_id: $agent_id")
            if run_id:
                source_props.append("run_id: $run_id")
                dest_props.append("run_id: $run_id")
            source_props_str = ", ".join(source_props)
            dest_props_str = ", ".join(dest_props)

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n {self.node_label} {{{source_props_str}}})
            -[r:{relationship}]->
            (m {self.node_label} {{{dest_props_str}}})
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results

    def _add_entities(self, to_be_added, filters, metadata=None):
        """Add the new entities to the graph. Merge the nodes if they already exist.
        
        Stores entity aliases and uses context-enhanced embeddings for better entity linking.
        Also stores temporal metadata (chapter, sequence) on relationships
        for cross-chapter causal reasoning.
        
        Args:
            to_be_added: List of entities to add, entity: (source, relationship, destination)
            filters: Filter conditions (user_id, agent_id, run_id)
            metadata: Temporal metadata:
                - chapter (int): Chapter number where this event occurs
                - sequence (int): Global sequence number for ordering events
        """
        metadata = metadata or {}
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        
        # Extract temporal info from metadata
        chapter = metadata.get("chapter")
        sequence = metadata.get("sequence")
        
        # For performance logging
        total_search_time = 0.0
        total_query_time = 0.0
        total_embedding_time = 0.0
        
        results = []
        for item in to_be_added:
            source_info = item.get("source", {})
            destination_info = item.get("destination", {})
            
            source = source_info.get("entity", "")
            destination = destination_info.get("entity", "") if destination_info else ""
            relationship = item.get("relationship", "")
            
            # Sanitize name for Cypher (replace special characters)
            if destination:
                destination = sanitize_relationship_for_cypher(destination.lower().replace(" ", "_"))
            if relationship:
                relationship = sanitize_relationship_for_cypher(relationship.lower().replace(" ", "_"))
            
            # Skip if source is empty
            if not source or not source.strip():
                continue
            else:
                source = sanitize_relationship_for_cypher(source.lower().replace(" ", "_"))

            # Get aliases and descriptions
            source_aliases = source_info.get("aliases", [])
            source_description = source_info.get("description", "")
            
            # types (now from entity_info_map)
            source_type = source_info.get("entity_type", "__User__")
            source_label = self.node_label if self.node_label else f":`{source_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""

            # Handle case: only source exists (no destination and relationship)
            if not destination or not destination.strip() or not relationship or not relationship.strip():
                # Only create/update source node
                t_search_start = time.perf_counter()
                source_node_search_result = self._search_source_node(
                    None, filters, threshold=self.threshold,
                    entity_name=source, aliases=source_aliases, skip_vector_search=True
                )
                if self.enable_perf_logging:
                    t_search_end = time.perf_counter()
                    total_search_time += (t_search_end - t_search_start)
                
                # Build source MERGE properties
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)
                
                if source_node_search_result:
                    # Update existing source node
                    cypher = f"""
                    MATCH (source)
                    WHERE elementId(source) = $source_id
                    SET source.mentions = coalesce(source.mentions, 0) + 1,
                        source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                        source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                    RETURN source.name AS source
                    """
                    params = {
                        "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                        "source_aliases": source_aliases,
                        "source_description": source_description,
                    }
                else:
                    # Create new source node
                    # self.enable_node_embeddings=True, compute embeddings
                    if self.enable_node_embeddings:
                        t_emb_start = time.perf_counter()
                        source_embedding = self._compute_enhanced_embedding(source, source_info)
                        if self.enable_perf_logging:
                            t_emb_end = time.perf_counter()
                            total_embedding_time += (t_emb_end - t_emb_start)
                        
                        cypher = f"""
                        MERGE (source {source_label} {{{merge_props_str}}})
                        ON CREATE SET source.created = timestamp(),
                                    source.mentions = 1,
                                    source.aliases = $source_aliases,
                                    source.description = $source_description
                                    {source_extra_set}
                        ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1,
                                    source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                                    source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                        WITH source
                        CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                        RETURN source.name AS source
                        """
                        params = {
                            "source_name": source,
                            "source_embedding": source_embedding,
                            "source_aliases": source_aliases,
                            "source_description": source_description,
                            "user_id": user_id,
                        }
                    else:
                        # self.enable_node_embeddings=False, do not compute embeddings
                        cypher = f"""
                        MERGE (source {source_label} {{{merge_props_str}}})
                        ON CREATE SET source.created = timestamp(),
                                    source.mentions = 1,
                                    source.aliases = $source_aliases,
                                    source.description = $source_description
                                    {source_extra_set}
                        ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1,
                                    source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                                    source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                        RETURN source.name AS source
                        """
                        params = {
                            "source_name": source,
                            "source_aliases": source_aliases,
                            "source_description": source_description,
                            "user_id": user_id,
                        }
                    if agent_id:
                        params["agent_id"] = agent_id
                    if run_id:
                        params["run_id"] = run_id
                
                t_query_start = time.perf_counter()
                result = self.graph.query(cypher, params=params)
                if self.enable_perf_logging:
                    t_query_end = time.perf_counter()
                    total_query_time += (t_query_end - t_query_start)
                
                logger.debug(f"Added source-only node: {result}")
                results.append(result)
                continue

            # Normal case: source, destination, and relationship all exist
            dest_aliases = destination_info.get("aliases", [])
            dest_description = destination_info.get("description", "")
            
            destination_type = destination_info.get("entity_type", "__User__")
            destination_label = self.node_label if self.node_label else f":`{destination_type}`"
            destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

            t_search_start = time.perf_counter()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                source_future = executor.submit(
                    self._search_source_node,
                    None, filters, self.threshold,
                    source, source_aliases, True  # skip_vector_search=True
                )
                dest_future = executor.submit(
                    self._search_destination_node,
                    None, filters, self.threshold,
                    destination, dest_aliases, True  # skip_vector_search=True
                )
                source_node_search_result = source_future.result()
                destination_node_search_result = dest_future.result()

            if self.enable_perf_logging:
                t_search_end = time.perf_counter()
                total_search_time += (t_search_end - t_search_start)

            # Based on the search results and the on/off switch, decide whether to compute the embedding
            source_embedding = None
            dest_embedding = None
            
            if self.enable_node_embeddings:
                if not source_node_search_result:
                    # A new source node needs to be created to calculate the embedding
                    t_emb_start = time.perf_counter()
                    source_embedding = self._compute_enhanced_embedding(source, source_info)
                    if self.enable_perf_logging:
                        total_embedding_time += (time.perf_counter() - t_emb_start)
                
                if not destination_node_search_result:
                    # A new destination node needs to be created to calculate the embedding
                    t_emb_start = time.perf_counter()
                    dest_embedding = self._compute_enhanced_embedding(destination, destination_info)
                    if self.enable_perf_logging:
                        total_embedding_time += (time.perf_counter() - t_emb_start)

            if not destination_node_search_result and source_node_search_result:
                # Build destination MERGE properties
                merge_props = ["name: $destination_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                embedding_clause = ""
                if self.enable_node_embeddings and dest_embedding is not None:
                    embedding_clause = """
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $destination_embedding)"""

                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1,
                    source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                WITH source
                MERGE (destination {destination_label} {{{merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1,
                    destination.aliases = $dest_aliases,
                    destination.description = $dest_description
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.aliases = reduce(seen = [], x IN coalesce(destination.aliases, []) + $dest_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    destination.description = CASE WHEN $dest_description <> '' THEN $dest_description ELSE destination.description END
                {embedding_clause}
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1,
                    r.chapter = $chapter,
                    r.sequence = $sequence
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1,
                    r.chapter = CASE WHEN $chapter IS NOT NULL THEN $chapter ELSE r.chapter END,
                    r.sequence = CASE WHEN $sequence IS NOT NULL THEN $sequence ELSE r.sequence END
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_name": destination,
                    "source_aliases": source_aliases,
                    "source_description": source_description,
                    "dest_aliases": dest_aliases,
                    "dest_description": dest_description,
                    "user_id": user_id,
                    "chapter": chapter,
                    "sequence": sequence,
                }
                if self.enable_node_embeddings and dest_embedding is not None:
                    params["destination_embedding"] = dest_embedding
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif destination_node_search_result and not source_node_search_result:
                # Build source MERGE properties
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                embedding_clause = ""
                if self.enable_node_embeddings and source_embedding is not None:
                    embedding_clause = """
                WITH source, destination
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)"""

                cypher = f"""
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.aliases = reduce(seen = [], x IN coalesce(destination.aliases, []) + $dest_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    destination.description = CASE WHEN $dest_description <> '' THEN $dest_description ELSE destination.description END
                WITH destination
                MERGE (source {source_label} {{{merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1,
                    source.aliases = $source_aliases,
                    source.description = $source_description
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1,
                    source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                {embedding_clause}
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1,
                    r.chapter = $chapter,
                    r.sequence = $sequence
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1,
                    r.chapter = CASE WHEN $chapter IS NOT NULL THEN $chapter ELSE r.chapter END,
                    r.sequence = CASE WHEN $sequence IS NOT NULL THEN $sequence ELSE r.sequence END
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_name": source,
                    "source_aliases": source_aliases,
                    "source_description": source_description,
                    "dest_aliases": dest_aliases,
                    "dest_description": dest_description,
                    "user_id": user_id,
                    "chapter": chapter,
                    "sequence": sequence,
                }
                if self.enable_node_embeddings and source_embedding is not None:
                    params["source_embedding"] = source_embedding
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1,
                    source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                WITH source
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.aliases = reduce(seen = [], x IN coalesce(destination.aliases, []) + $dest_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    destination.description = CASE WHEN $dest_description <> '' THEN $dest_description ELSE destination.description END
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1,
                    r.chapter = $chapter,
                    r.sequence = $sequence
                ON MATCH SET 
                    r.mentions = coalesce(r.mentions, 0) + 1,
                    r.chapter = CASE WHEN $chapter IS NOT NULL THEN $chapter ELSE r.chapter END,
                    r.sequence = CASE WHEN $sequence IS NOT NULL THEN $sequence ELSE r.sequence END
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_aliases": source_aliases,
                    "source_description": source_description,
                    "dest_aliases": dest_aliases,
                    "dest_description": dest_description,
                    "user_id": user_id,
                    "chapter": chapter,
                    "sequence": sequence,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            else:
                # Build dynamic MERGE props for both source and destination
                source_props = ["name: $source_name", "user_id: $user_id"]
                dest_props = ["name: $dest_name", "user_id: $user_id"]
                if agent_id:
                    source_props.append("agent_id: $agent_id")
                    dest_props.append("agent_id: $agent_id")
                if run_id:
                    source_props.append("run_id: $run_id")
                    dest_props.append("run_id: $run_id")
                source_props_str = ", ".join(source_props)
                dest_props_str = ", ".join(dest_props)

                source_emb_clause = ""
                dest_emb_clause = ""
                if self.enable_node_embeddings and source_embedding is not None:
                    source_emb_clause = """
                WITH source
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)"""
                if self.enable_node_embeddings and dest_embedding is not None:
                    dest_emb_clause = """
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $dest_embedding)"""

                cypher = f"""
                MERGE (source {source_label} {{{source_props_str}}})
                ON CREATE SET source.created = timestamp(),
                            source.mentions = 1,
                            source.aliases = $source_aliases,
                            source.description = $source_description
                            {source_extra_set}
                ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1,
                            source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                            source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                {source_emb_clause}
                WITH source
                MERGE (destination {destination_label} {{{dest_props_str}}})
                ON CREATE SET destination.created = timestamp(),
                            destination.mentions = 1,
                            destination.aliases = $dest_aliases,
                            destination.description = $dest_description
                            {destination_extra_set}
                ON MATCH SET destination.mentions = coalesce(destination.mentions, 0) + 1,
                            destination.aliases = reduce(seen = [], x IN coalesce(destination.aliases, []) + $dest_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                            destination.description = CASE WHEN $dest_description <> '' THEN $dest_description ELSE destination.description END
                {dest_emb_clause}
                WITH source, destination
                MERGE (source)-[rel:{relationship}]->(destination)
                ON CREATE SET 
                    rel.created = timestamp(), 
                    rel.mentions = 1,
                    rel.chapter = $chapter,
                    rel.sequence = $sequence
                ON MATCH SET 
                    rel.mentions = coalesce(rel.mentions, 0) + 1,
                    rel.chapter = CASE WHEN $chapter IS NOT NULL THEN $chapter ELSE rel.chapter END,
                    rel.sequence = CASE WHEN $sequence IS NOT NULL THEN $sequence ELSE rel.sequence END
                RETURN source.name AS source, type(rel) AS relationship, destination.name AS target
                """

                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_aliases": source_aliases,
                    "source_description": source_description,
                    "dest_aliases": dest_aliases,
                    "dest_description": dest_description,
                    "user_id": user_id,
                    "chapter": chapter,
                    "sequence": sequence,
                }
                if self.enable_node_embeddings and source_embedding is not None:
                    params["source_embedding"] = source_embedding
                if self.enable_node_embeddings and dest_embedding is not None:
                    params["dest_embedding"] = dest_embedding
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id
            t_query_start = time.perf_counter()
            result = self.graph.query(cypher, params=params)
            if self.enable_perf_logging:
                t_query_end = time.perf_counter()
                total_query_time += (t_query_end - t_query_start)
            
            logger.debug(f"Added relationships: {result}")
            results.append(result)
        
        # Print logging
        if self.enable_perf_logging:
            logger.info(f"[_add_entities 计时] Embedding 计算: {total_embedding_time:.3f}s")
            logger.info(f"[_add_entities 计时] 节点搜索总耗时: {total_search_time:.3f}s")
            logger.info(f"[_add_entities 计时] Neo4j 语句总耗时: {total_query_time:.3f}s")
            logger.info(f"[_add_entities 计时] 处理实体数量: {len(to_be_added)}")
        
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            if item.get("source", "").strip() == "":
                continue
            item["source"] = item["source"].lower().replace(" ", "_")
            # Use the sanitization function for relationships to handle special characters
            item["relationship"] = sanitize_relationship_for_cypher(item["relationship"].lower().replace(" ", "_"))
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9, entity_name=None, aliases=None,
                            skip_vector_search=False):
        """Search for existing source node by name/alias matching first, then by embedding similarity.
        
        Strategy：
        1. Try exact name matching first (fastest).
        2. If not found and skip_vector_search=False, then try vector search.
        
        Args:
            source_embedding: The embedding vector of the source entity
            filters: Filter conditions (user_id, agent_id, run_id)
            threshold: Similarity threshold for embedding matching
            entity_name: The name of the entity to match against node names and aliases
            aliases: List of aliases to check for matches
            skip_vector_search: If True, skip expensive vector search (for new nodes)
        """
        # Build filter params
        params = {
            "user_id": filters["user_id"],
            "entity_name": entity_name,
            "aliases": aliases or [],
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        
        # Build WHERE conditions for user filtering
        user_conditions = ["n.user_id = $user_id"]
        if filters.get("agent_id"):
            user_conditions.append("n.agent_id = $agent_id")
        if filters.get("run_id"):
            user_conditions.append("n.run_id = $run_id")
        user_filter = " AND ".join(user_conditions)

        # Step 1: exact name matching
        fast_cypher = f"""
            MATCH (n {self.node_label})
            WHERE {user_filter}
              AND (n.name = $entity_name 
                   OR $entity_name IN coalesce(n.aliases, [])
                   OR size([a IN $aliases WHERE a = n.name OR a IN coalesce(n.aliases, [])]) > 0)
            RETURN elementId(n) AS `elementId(source_candidate)`
            LIMIT 1
            """
        
        result = self.graph.query(fast_cypher, params=params)
        if result:
            return result
        
        # Step 2: if not found and skip_vector_search=False, then try vector search
        if skip_vector_search or source_embedding is None:
            return []
            
        params["source_embedding"] = source_embedding
        params["threshold"] = threshold
        
        vector_cypher = f"""
            MATCH (n {self.node_label})
            WHERE {user_filter} AND n.embedding IS NOT NULL
            WITH n, vector.similarity.cosine(n.embedding, $source_embedding) AS similarity
            WHERE similarity >= $threshold
            ORDER BY similarity DESC
            LIMIT 1
            RETURN elementId(n) AS `elementId(source_candidate)`
            """

        result = self.graph.query(vector_cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9, entity_name=None, aliases=None, skip_vector_search=False):
        """Search for existing destination node by name/alias matching first, then by embedding similarity.
        
        Strategy：
        1. Try exact name matching first (fastest).
        2. If not found and skip_vector_search=False, then try vector search.
        
        Args:
            destination_embedding: The embedding vector of the destination entity
            filters: Filter conditions (user_id, agent_id, run_id)
            threshold: Similarity threshold for embedding matching
            entity_name: The name of the entity to match against node names and aliases
            aliases: List of aliases to check for matches
            skip_vector_search: If True, skip expensive vector search (for new nodes)
        """
        # Build filter params
        params = {
            "user_id": filters["user_id"],
            "entity_name": entity_name,
            "aliases": aliases or [],
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        
        # Build WHERE conditions for user filtering
        user_conditions = ["n.user_id = $user_id"]
        if filters.get("agent_id"):
            user_conditions.append("n.agent_id = $agent_id")
        if filters.get("run_id"):
            user_conditions.append("n.run_id = $run_id")
        user_filter = " AND ".join(user_conditions)

        # Step 1: exact name matching
        fast_cypher = f"""
            MATCH (n {self.node_label})
            WHERE {user_filter}
              AND (n.name = $entity_name 
                   OR $entity_name IN coalesce(n.aliases, [])
                   OR size([a IN $aliases WHERE a = n.name OR a IN coalesce(n.aliases, [])]) > 0)
            RETURN elementId(n) AS `elementId(destination_candidate)`
            LIMIT 1
            """
        
        result = self.graph.query(fast_cypher, params=params)
        if result:
            return result
        
        # Step 2: if not found and skip_vector_search=False, then try vector search
        if skip_vector_search or destination_embedding is None:
            return []
            
        params["destination_embedding"] = destination_embedding
        params["threshold"] = threshold
        
        vector_cypher = f"""
            MATCH (n {self.node_label})
            WHERE {user_filter} AND n.embedding IS NOT NULL
            WITH n, vector.similarity.cosine(n.embedding, $destination_embedding) AS similarity
            WHERE similarity >= $threshold
            ORDER BY similarity DESC
            LIMIT 1
            RETURN elementId(n) AS `elementId(destination_candidate)`
            """

        result = self.graph.query(vector_cypher, params=params)
        return result

    # Reset is not defined in base.py
    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
