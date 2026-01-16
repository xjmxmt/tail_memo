# TailMemo Backend

基于 FastAPI 的人物关系图谱与记忆查询后端服务。

## 安装依赖

```bash
cd visualization/backend

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 Windows:
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 本地运行

```bash
# 开发模式（热重载）
uvicorn app.main:app --reload --port 8000

# 或使用 Python 直接运行
python -m uvicorn app.main:app --reload --port 8000
```

服务启动后访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## Docker 运行

```bash
cd visualization

# 启动所有服务（PostgreSQL + Neo4j + Backend）
docker-compose up -d

# 查看日志
docker-compose logs -f backend

# 停止服务
docker-compose down
```

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 服务状态 |
| GET | `/health` | 健康检查 |
| POST | `/api/query` | 查询记忆 |
| POST | `/api/query/postgres` | 从 PostgreSQL 查询 |
| POST | `/api/query/neo4j` | 从 Neo4j 查询 |
| GET | `/api/graph` | 获取人物关系图 |
| POST | `/api/graph/node` | 添加人物节点 |
| POST | `/api/graph/edge` | 添加关系边 |

## 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `POSTGRES_HOST` | localhost | PostgreSQL 主机 |
| `POSTGRES_PORT` | 5432 | PostgreSQL 端口 |
| `POSTGRES_USER` | myuser | PostgreSQL 用户 |
| `POSTGRES_PASSWORD` | mypassword | PostgreSQL 密码 |
| `POSTGRES_DB` | mydatabase | PostgreSQL 数据库 |
| `NEO4J_URI` | bolt://localhost:7687 | Neo4j 连接 URI |
| `NEO4J_USER` | neo4j | Neo4j 用户 |
| `NEO4J_PASSWORD` | neo4jpassword | Neo4j 密码 |

## 项目结构

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 应用入口
│   ├── config.py        # 配置管理
│   ├── database.py      # 数据库连接
│   ├── schemas.py       # Pydantic 模型
│   └── routes/
│       ├── __init__.py
│       ├── query.py     # 记忆查询路由
│       └── graph.py     # 图谱数据路由
├── Dockerfile
├── requirements.txt
└── README.md
```

