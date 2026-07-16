# 🏢 企业知识库智能问答 Agent

基于 **RAG + LangGraph** 的企业文档智能问答系统。上传 PDF/Word/TXT 文档，即可进行多轮知识问答，回答附带引用来源。

## 🎯 核心能力

| 能力 | 实现 |
|------|------|
| 📄 多格式解析 | PDF（含表格）、Word、TXT、Markdown |
| ✂️ 智能分块 | 递归语义分块，保留上下文 |
| 🔍 混合检索 | 向量检索 + BGE Reranker 重排序 |
| 🤖 Agent 工作流 | LangGraph 多步骤：检索 → 验证 → 生成 |
| 💬 多轮对话 | 会话记忆，上下文关联 |
| 📌 引用溯源 | 每个答案标注来源文档和段落 |
| 👍 用户反馈 | 支持点赞/踩，持续优化 |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

获取 API Key：https://platform.deepseek.com

### 3. 启动服务

```bash
uvicorn app.main:app --reload
```

### 4. 访问

- **API 文档**：http://localhost:8000/docs
- **交互式测试**：直接在 /docs 页面测试所有接口

## 📡 API 接口

### 文档管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/documents/upload` | 上传文档 |
| GET | `/documents/` | 文档列表 |
| DELETE | `/documents/{id}` | 删除文档 |

### 智能问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 知识问答（非流式） |
| POST | `/chat/stream` | 知识问答（流式 SSE） |
| POST | `/chat/{msg_id}/feedback` | 提交反馈 |

### 会话管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/sessions` | 创建会话 |
| GET | `/sessions` | 会话列表 |
| DELETE | `/sessions/{id}` | 删除会话 |

## 🐳 Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 生产环境（含 MySQL）
docker-compose --profile production up -d
```

## 🏗️ 技术架构

```
用户请求
  ↓
FastAPI 路由层
  ↓
LangGraph Agent 工作流
  ├── ① 检索（Retrieve）
  │     ├── Embedding 向量化
  │     └── ChromaDB 相似度检索
  ├── ② 验证（Verify）
  │     └── 检索质量检查
  └── ③ 生成（Generate）
        ├── 构建 Context（检索结果 + 历史）
        ├── LLM 生成回答 + 引用标注
        └── 保存消息 + 返回结果

数据层：
  向量库 ← ChromaDB
  关系库 ← SQLite / MySQL（SQLAlchemy）
```

## 📁 项目结构

```
enterprise-rag-agent/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 全局配置
│   ├── api/
│   │   ├── documents.py     # 文档管理 API
│   │   └── chat.py          # 问答 + 会话 API
│   ├── core/
│   │   ├── document_parser.py  # 多格式文档解析
│   │   ├── embedding.py        # Embedding 服务
│   │   ├── vector_store.py     # ChromaDB 操作
│   │   ├── retriever.py        # 混合检索 + Reranker
│   │   ├── generator.py        # LLM 答案生成
│   │   └── agent.py            # LangGraph Agent
│   ├── models/
│   │   ├── database.py         # 数据库引擎
│   │   ├── document.py         # 文档元数据模型
│   │   └── session.py          # 会话 & 消息模型
│   └── schemas/
│       └── __init__.py         # Pydantic 请求/响应模型
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## ⚙️ 配置说明

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_API_KEY` | — | DeepSeek API Key |
| `LLM_MODEL` | deepseek-chat | LLM 模型 |
| `DATABASE_URL` | sqlite:///app.db | 数据库地址 |
| `CHUNK_SIZE` | 800 | 分块大小 |
| `RETRIEVAL_TOP_K` | 10 | 检索返回数 |
| `USE_RERANKER` | true | 启用重排序 |

## 📝 License

MIT
