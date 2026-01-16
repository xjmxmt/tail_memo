from fastapi import APIRouter
from ..schemas import QueryRequest, QueryResponse, MemoryItem
from ..database import get_pg_connection, get_neo4j_session, pg_pool, neo4j_driver

router = APIRouter(prefix="/api", tags=["query"])


# 硬编码的模拟数据（后续替换为真实数据库查询）
MOCK_DATA: dict[str, dict] = {
    "李清月": {
        "reply": "找到了关于李清月的相关记忆",
        "memories": [
            {"content": "李清月是故事的主角，性格坚韧果敢", "source": "角色设定"},
            {"content": "李清月与林婉儿是母女关系，感情深厚", "source": "关系图谱"},
            {"content": "李清月与顾兵结为盟友，共同对抗敌人", "source": "剧情记录"},
        ],
    },
    "林婉儿": {
        "reply": "找到了关于林婉儿的相关记忆",
        "memories": [
            {"content": "林婉儿是李清月的女儿", "source": "关系图谱"},
            {"content": "林婉儿性格温柔善良，但内心坚强", "source": "角色设定"},
        ],
    },
    "顾兵": {
        "reply": "找到了关于顾兵的相关记忆",
        "memories": [
            {"content": "顾兵是李清月的盟友", "source": "关系图谱"},
            {"content": "顾兵武艺高强，忠诚可靠", "source": "角色设定"},
        ],
    },
}


@router.post("/query", response_model=QueryResponse)
async def query_memory(request: QueryRequest):
    """
    查询记忆
    
    根据输入的关键词查询相关记忆。
    目前使用硬编码数据，后续可替换为真实数据库查询。
    """
    query = request.query.strip()
    
    # 查找匹配的数据
    matched_key = None
    for key in MOCK_DATA.keys():
        if key in query:
            matched_key = key
            break
    
    if matched_key:
        data = MOCK_DATA[matched_key]
        return QueryResponse(
            reply=data["reply"],
            memories=[
                MemoryItem(
                    content=m["content"],
                    source=m.get("source"),
                    similarity=m.get("similarity"),
                )
                for m in data["memories"]
            ],
        )
    
    # 未找到匹配
    return QueryResponse(
        reply="暂未找到相关记忆，请尝试其他查询",
        memories=[],
    )


@router.post("/query/postgres")
async def query_from_postgres(request: QueryRequest):
    """
    从 PostgreSQL 查询记忆（示例）
    
    使用 pgvector 进行语义搜索。
    需要先在数据库中创建相应的表和向量索引。
    """
    if not pg_pool:
        return QueryResponse(
            reply="数据库未连接",
            memories=[],
        )
    
    async with get_pg_connection() as conn:
        # 示例查询 - 需要根据实际表结构修改
        # rows = await conn.fetch(
        #     """
        #     SELECT content, source, 1 - (embedding <=> $1::vector) as similarity
        #     FROM memories
        #     ORDER BY embedding <=> $1::vector
        #     LIMIT 5
        #     """,
        #     query_embedding,  # 需要先将查询文本转换为向量
        # )
        
        # 目前返回示例数据
        return QueryResponse(
            reply=f"PostgreSQL 查询: {request.query}",
            memories=[
                MemoryItem(content="这是一条来自 PostgreSQL 的示例记忆", source="postgres"),
            ],
        )


@router.post("/query/neo4j")
async def query_from_neo4j(request: QueryRequest):
    """
    从 Neo4j 查询关系（示例）
    
    查询人物之间的关系。
    """
    if not neo4j_driver:
        return QueryResponse(
            reply="Neo4j 未连接",
            memories=[],
        )
    
    async with get_neo4j_session() as session:
        # 示例查询 - 查找与某个人物相关的所有关系
        # result = await session.run(
        #     """
        #     MATCH (p:Person {name: $name})-[r]->(other)
        #     RETURN p.name as source, type(r) as relation, other.name as target
        #     LIMIT 10
        #     """,
        #     name=request.query,
        # )
        # records = await result.data()
        
        # 目前返回示例数据
        return QueryResponse(
            reply=f"Neo4j 查询: {request.query}",
            memories=[
                MemoryItem(content="这是一条来自 Neo4j 的示例关系", source="neo4j"),
            ],
        )

