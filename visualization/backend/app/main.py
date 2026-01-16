from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_postgres, close_postgres, init_neo4j, close_neo4j
from .routes import query, graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化数据库连接
    print("🚀 启动 TailMemo 后端服务...")
    try:
        await init_postgres()
    except Exception as e:
        print(f"⚠️ PostgreSQL 连接失败: {e}")
    
    try:
        init_neo4j()
    except Exception as e:
        print(f"⚠️ Neo4j 连接失败: {e}")
    
    yield
    
    # 关闭时清理资源
    print("👋 关闭 TailMemo 后端服务...")
    await close_postgres()
    await close_neo4j()


app = FastAPI(
    title="TailMemo API",
    description="人物关系图谱与记忆查询后端服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(query.router)
app.include_router(graph.router)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "service": "TailMemo API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

