UPDATE_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "update_graph_memory",
        "description": "Update the relationship key of an existing graph memory based on new information. This function should be called when there's a need to modify an existing relationship in the knowledge graph. The update should only be performed if the new information is more recent, more accurate, or provides additional context compared to the existing information. The source and destination nodes of the relationship must remain the same as in the existing graph memory; only the relationship itself can be updated.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The new or updated relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
            },
            "required": ["source", "destination", "relationship"],
            "additionalProperties": False,
        },
    },
}

ADD_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "add_graph_memory",
        "description": "Add a new graph memory to the knowledge graph. This function creates a new relationship between two nodes, potentially creating new nodes if they don't exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The type of relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
                "source_type": {
                    "type": "string",
                    "description": "The type or category of the source node. This helps in classifying and organizing nodes in the graph.",
                },
                "destination_type": {
                    "type": "string",
                    "description": "The type or category of the destination node. This helps in classifying and organizing nodes in the graph.",
                },
            },
            "required": [
                "source",
                "destination",
                "relationship",
                "source_type",
                "destination_type",
            ],
            "additionalProperties": False,
        },
    },
}


NOOP_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "No operation should be performed to the graph entities. This function is called when the system determines that no changes or additions are necessary based on the current input or context. It serves as a placeholder action when no other actions are required, ensuring that the system can explicitly acknowledge situations where no modifications to the graph are needed.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relationships",
        "description": "根据输入文本提取实体，如有则同时提取实体及其关系",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string", "description": "实体的名称"},
                                    "entity_type": {"type": "string", "description": "实体的类型或类别"},
                                    "description": {"type": "string", "description": "关系中的源实体，"
                                                                                     "基于上下文的简要描述，包括文本中提到的关键属性、角色或特征"},
                                    "aliases": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "在文本中用来指代此实体的其他名称、昵称或引用",
                                    }
                                }
                            },
                            "relationship": {
                                "type": "string",
                                "description": "源实体与目标实体之间的关系",
                            },
                            "destination": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string", "description": "实体的名称"},
                                    "entity_type": {"type": "string", "description": "实体的类型或类别"},
                                    "description": {"type": "string", "description": "关系中的目标实体，"
                                                                                     "基于上下文的简要描述，包括文本中提到的关键属性、角色或特征"},
                                    "aliases": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "在文本中用来指代此实体的其他名称、昵称或引用",
                                    }
                                }
                            },
                        },
                        "required": [
                            "source"
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}


EXTRACT_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "从文本中提取实体名称及其类型，包括简要描述和别名",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "实体的名称"},
                            "entity_type": {"type": "string", "description": "实体的类型或类别"},
                            "description": {
                                "type": "string",
                                "description": "基于上下文的简要描述，包括文本中提到的关键属性、角色或特征",
                            },
                            "aliases": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "在文本中用来指代此实体的其他名称、昵称或引用",
                            },
                        },
                        "required": ["entity", "entity_type"],
                        "additionalProperties": False,
                    },
                    "description": "包含实体类型、描述和别名的实体列表",
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

UPDATE_MEMORY_STRUCT_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "update_graph_memory",
        "description": "Update the relationship key of an existing graph memory based on new information. This function should be called when there's a need to modify an existing relationship in the knowledge graph. The update should only be performed if the new information is more recent, more accurate, or provides additional context compared to the existing information. The source and destination nodes of the relationship must remain the same as in the existing graph memory; only the relationship itself can be updated.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The new or updated relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
            },
            "required": ["source", "destination", "relationship"],
            "additionalProperties": False,
        },
    },
}

ADD_MEMORY_STRUCT_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "add_graph_memory",
        "description": "Add a new graph memory to the knowledge graph. This function creates a new relationship between two nodes, potentially creating new nodes if they don't exist.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The type of relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
                "source_type": {
                    "type": "string",
                    "description": "The type or category of the source node. This helps in classifying and organizing nodes in the graph.",
                },
                "destination_type": {
                    "type": "string",
                    "description": "The type or category of the destination node. This helps in classifying and organizing nodes in the graph.",
                },
            },
            "required": [
                "source",
                "destination",
                "relationship",
                "source_type",
                "destination_type",
            ],
            "additionalProperties": False,
        },
    },
}


NOOP_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "No operation should be performed to the graph entities. This function is called when the system determines that no changes or additions are necessary based on the current input or context. It serves as a placeholder action when no other actions are required, ensuring that the system can explicitly acknowledge situations where no modifications to the graph are needed.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

RELATIONS_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relations",
        "description": "Establish relationships among the entities based on the provided text.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "The source entity of the relationship.",
                            },
                            "relationship": {
                                "type": "string",
                                "description": "The relationship between the source and destination entities.",
                            },
                            "destination": {
                                "type": "string",
                                "description": "The destination entity of the relationship.",
                            },
                        },
                        "required": [
                            "source",
                            "relationship",
                            "destination",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}


EXTRACT_ENTITIES_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract entities and their types from the text, along with a brief description and any known aliases.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "The canonical/primary name of the entity."},
                            "entity_type": {"type": "string", "description": "The type or category of the entity."},
                            "description": {
                                "type": "string",
                                "description": "A brief description of the entity based on the context, including key attributes, roles, or characteristics mentioned in the text.",
                            },
                            "aliases": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Alternative names, nicknames, or references used to refer to this entity in the text.",
                            },
                        },
                        "required": ["entity", "entity_type", "description", "aliases"],
                        "additionalProperties": False,
                    },
                    "description": "An array of entities with their types, descriptions, and aliases.",
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

DELETE_MEMORY_STRUCT_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "delete_graph_memory",
        "description": "Delete the relationship between two nodes. This function deletes the existing relationship.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the relationship.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The existing relationship between the source and destination nodes that needs to be deleted.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the relationship.",
                },
            },
            "required": [
                "source",
                "relationship",
                "destination",
            ],
            "additionalProperties": False,
        },
    },
}

DELETE_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "delete_graph_memory",
        "description": "Delete the relationship between two nodes. This function deletes the existing relationship.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the relationship.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The existing relationship between the source and destination nodes that needs to be deleted.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the relationship.",
                },
            },
            "required": [
                "source",
                "relationship",
                "destination",
            ],
            "additionalProperties": False,
        },
    },
}

# ==================== Causal Reasoning Tools ====================

CAUSAL_REASONING_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "decide_next_action",
        "description": "根据当前检索到的图谱信息，决定下一步操作：继续探索因果链、完成分析或返回当前结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["continue", "complete"],
                    "description": "下一步行动：'continue' 继续探索相关节点，'complete' 已获得足够信息完成分析",
                },
                "next_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "如果 action 为 'continue'，指定要继续探索的实体名称列表",
                },
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "both"],
                    "description": "探索方向：'forward' 沿时间线向后探索后续事件，'backward' 向前探索原因，'both' 双向探索",
                },
                "reasoning": {
                    "type": "string",
                    "description": "简要说明做出此决定的原因",
                },
                "answer": {
                    "type": "string",
                    "description": "如果 action 为 'complete'，请将得到的信息整理成问题的回答",
                }
            },
            "required": ["action", "reasoning"],
            "additionalProperties": False,
        },
    },
}

CAUSAL_REASONING_DECISION_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "decide_next_action",
        "description": "根据当前检索到的图谱信息，决定下一步操作：继续探索因果链、完成分析或返回当前结果。",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["continue", "complete"],
                    "description": "下一步行动：'continue' 继续探索相关节点，'complete' 已获得足够信息完成分析",
                },
                "next_entities": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "如果 action 为 'continue'，指定要继续探索的实体名称列表",
                },
                "direction": {
                    "type": ["string", "null"],
                    "enum": ["forward", "backward", "both", None],
                    "description": "探索方向：'forward' 沿时间线向后探索后续事件，'backward' 向前探索原因，'both' 双向探索",
                },
                "reasoning": {
                    "type": "string",
                    "description": "简要说明做出此决定的原因",
                },
            },
            "required": ["action", "next_entities", "direction", "reasoning"],
            "additionalProperties": False,
        },
    },
}

