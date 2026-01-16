from pydantic import BaseModel


class QueryRequest(BaseModel):
    """查询请求"""
    query: str


class MemoryItem(BaseModel):
    """记忆条目"""
    content: str
    source: str | None = None
    similarity: float | None = None


class QueryResponse(BaseModel):
    """查询响应"""
    reply: str
    memories: list[MemoryItem]


class CharacterNode(BaseModel):
    """人物节点"""
    id: str
    name: str
    type: str = "sub"  # main, sub, minor
    color: str = "bg-white"


class RelationshipEdge(BaseModel):
    """关系边"""
    id: str
    source: str
    target: str
    label: str


class GraphData(BaseModel):
    """图数据"""
    nodes: list[CharacterNode]
    edges: list[RelationshipEdge]

