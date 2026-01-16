from fastapi import APIRouter
from ..schemas import GraphData, CharacterNode, RelationshipEdge

router = APIRouter(prefix="/api", tags=["graph"])


# 示例图数据
SAMPLE_GRAPH = GraphData(
    nodes=[
        CharacterNode(id="1", name="李清月", type="main", color="bg-white"),
        CharacterNode(id="2", name="林婉儿", type="sub", color="bg-white"),
        CharacterNode(id="3", name="顾兵", type="sub", color="bg-white"),
    ],
    edges=[
        RelationshipEdge(id="e1-2", source="1", target="2", label="母女"),
        RelationshipEdge(id="e1-3", source="1", target="3", label="盟友"),
    ],
)


@router.get("/graph", response_model=GraphData)
async def get_graph():
    """
    获取人物关系图数据
    
    返回所有人物节点和关系边。
    后续可替换为从 Neo4j 查询。
    """
    return SAMPLE_GRAPH


@router.post("/graph/node")
async def add_node(node: CharacterNode):
    """添加人物节点"""
    # TODO: 保存到 Neo4j
    return {"status": "ok", "node": node}


@router.post("/graph/edge")
async def add_edge(edge: RelationshipEdge):
    """添加关系边"""
    # TODO: 保存到 Neo4j
    return {"status": "ok", "edge": edge}

