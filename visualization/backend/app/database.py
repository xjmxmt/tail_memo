import asyncpg
from neo4j import AsyncGraphDatabase
from contextlib import asynccontextmanager
from .config import settings

# PostgreSQL 连接池
pg_pool: asyncpg.Pool | None = None

# Neo4j 驱动
neo4j_driver = None


async def init_postgres():
    """初始化 PostgreSQL 连接池"""
    global pg_pool
    pg_pool = await asyncpg.create_pool(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        database=settings.POSTGRES_DB,
        min_size=2,
        max_size=10,
    )
    print("✅ PostgreSQL 连接池已创建")


async def close_postgres():
    """关闭 PostgreSQL 连接池"""
    global pg_pool
    if pg_pool:
        await pg_pool.close()
        print("PostgreSQL 连接池已关闭")


def init_neo4j():
    """初始化 Neo4j 驱动"""
    global neo4j_driver
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    print("✅ Neo4j 驱动已创建")


async def close_neo4j():
    """关闭 Neo4j 驱动"""
    global neo4j_driver
    if neo4j_driver:
        await neo4j_driver.close()
        print("Neo4j 驱动已关闭")


@asynccontextmanager
async def get_pg_connection():
    """获取 PostgreSQL 连接"""
    async with pg_pool.acquire() as conn:
        yield conn


@asynccontextmanager
async def get_neo4j_session():
    """获取 Neo4j 会话"""
    async with neo4j_driver.session() as session:
        yield session

