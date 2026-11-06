# AI Docs RAG - 技术文档

## 项目概述

**AI Docs RAG** 是一个基于 Agentic RAG（智能体检索增强生成）的 AI 技术文档问答系统。系统能够智能地检索、评估和生成关于 AI/ML 技术文档的回答，并提供完整的推理过程透明性。

### 核心特性

- **Agentic RAG**: 6 节点 LangGraph 工作流，具备自主决策能力
- **混合搜索**: BM25 + 向量搜索，通过 RRF（Reciprocal Rank Fusion）融合
- **智能评估**: 自动评估检索文档的相关性
- **查询重写**: 当检索结果不相关时自动优化查询
- **范围防护**: Guardrail 机制确保问题在专业领域内
- **可观测性**: 集成 Langfuse 进行全链路追踪

---

## 系统架构

### 技术栈

```
┌─────────────────────────────────────────────────────────┐
│                      API Layer                           │
│  FastAPI + Pydantic + Dependency Injection              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Agent Core (LangGraph)                  │
│  6-Node StateGraph: Guardrail → Retrieve → Grade →     │
│                     Rewrite → Generate → Out-of-Scope   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌──────────────┬──────────────┬──────────────┐
│  OpenSearch  │    Ollama    │  Embeddings  │
│  (向量数据库) │   (LLM)      │  (百炼/Ollama) │
└──────────────┴──────────────┴──────────────┘
                          ↓
                    ┌──────────┐
                    │  Redis   │
                    │  (缓存)   │
                    └──────────┘
```

### 核心组件

| 组件 | 技术 | 作用 |
|------|------|------|
| **API 层** | FastAPI | RESTful API，依赖注入，自动文档 |
| **Agent 核心** | LangGraph | 6 节点状态图，实现 Agentic 工作流 |
| **向量数据库** | OpenSearch 2.19 | 存储文档块，支持混合搜索 |
| **LLM** | Ollama (qwen2.5:7b) | 本地部署的大语言模型 |
| **Embedding** | 阿里百炼 / Ollama | 文本向量化 |
| **缓存** | Redis 7 | 查询结果缓存，TTL 6 小时 |
| **可观测性** | Langfuse | 全链路追踪，性能监控 |

---

## Agentic RAG 工作流

### 6 节点状态图

```
┌──────────────┐
│   用户提问    │
└──────┬───────┘
       ↓
┌──────────────────┐
│ 1. Guardrail 节点 │ ← 验证问题是否在 AI 技术领域内
└──────┬───────────┘
       ↓
   ┌───┴───┐
   │ 通过？ │
   └───┬───┘
       │ 是
       ↓
┌──────────────────┐
│ 2. Retrieve 节点  │ ← 调用检索工具获取相关文档块
└──────┬───────────┘
       ↓
┌──────────────────┐
│ 3. Grade 节点     │ ← LLM 评估每个文档的相关性
└──────┬───────────┘
       ↓
   ┌───┴───┐
   │相关？  │
   └───┬───┘
    是 │ 否（且重试次数 < 2）
       │        ↓
       │  ┌──────────────────┐
       │  │ 4. Rewrite 节点   │ ← 重写查询，重新检索
       │  └──────┬───────────┘
       │         │
       └────┬────┘
            ↓
     ┌──────┴──────┐
     │ 有相关文档？  │
     └──────┬──────┘
       是   │   否
       ↓    │    ↓
┌──────────┐ │ ┌──────────────────┐
│ 5. Generate│ │ │ 6. Out-of-Scope  │
│   节点    │ │ │    节点           │
└──────────┘ │ └──────────────────┘
             ↓
        拒绝回答
```

### 节点详解

#### 1. Guardrail 节点
**作用**: 过滤非 AI 技术领域的问题

**实现**:
- LLM 对问题进行评分（0-100）
- 阈值：60 分
- 低于阈值直接拒绝回答

**示例**:
```python
# 评分标准
"什么是 Transformer？" → 85 分 ✓
"今天天气怎么样？" → 20 分 ✗
```

#### 2. Retrieve 节点
**作用**: 检索相关文档块

**实现**:
- 调用 `retrieve_documents` 工具
- 使用混合搜索（BM25 + 向量）
- 返回 top_k 个文档块（默认 3）

**工具定义**:
```python
@tool
def retrieve_documents(query: str, top_k: int = 3) -> List[Dict]:
    """检索 AI 技术文档块"""
    # 混合搜索逻辑
    return results
```

#### 3. Grade 节点
**作用**: 评估检索文档的相关性

**实现**:
- LLM 对每个文档块评分（0-10）
- 阈值：7 分
- 返回相关文档列表

**输出**:
```python
{
    "relevant_sources": [...],  # 相关文档
    "grading_results": [...]    # 评分详情
}
```

#### 4. Rewrite 节点
**作用**: 当文档不相关时重写查询

**触发条件**:
- 所有文档评分 < 7
- 重试次数 < 2

**实现**:
- LLM 生成新的查询（更具体/更宽泛）
- 重新进入 Retrieve 节点

#### 5. Generate 节点
**作用**: 基于相关文档生成回答

**实现**:
- 使用相关文档作为上下文
- LLM 生成带引用的回答
- 提取来源信息

**输出**:
```json
{
  "answer": "Transformer 是一种...",
  "sources": ["doc1.pdf", "doc2.pdf"],
  "reasoning_steps": [...]
}
```

#### 6. Out-of-Scope 节点
**作用**: 当无法找到相关文档时拒绝回答

**触发条件**:
- 重试 2 次后仍无相关文档
- 或 Guardrail 评分过低

---

## 混合搜索实现

### 搜索流程

```
用户查询
   ↓
┌────────────────┐
│ 1. BM25 搜索    │ ← 关键词匹配，返回 top_k * 2
└────────┬───────┘
         ↓
┌────────────────┐
│ 2. 向量搜索     │ ← 语义相似度，返回 top_k * 2
└────────┬───────┘
         ↓
┌────────────────┐
│ 3. RRF 融合     │ ← Reciprocal Rank Fusion
└────────┬───────┘
         ↓
    返回 top_k 结果
```

### RRF 算法

**公式**:
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

其中：
- `d`: 文档
- `k`: 常数（通常 60）
- `rank_i(d)`: 文档在第 i 个检索器中的排名

**优势**:
- 平衡关键词匹配和语义相似度
- 避免单一检索器的偏差
- 提高召回率

### OpenSearch 配置

```python
# 索引映射
{
    "mappings": {
        "properties": {
            "chunk_text": {"type": "text"},
            "chunk_embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "space_type": "cosinesimil"
            },
            "doc_id": {"type": "keyword"},
            "section_name": {"type": "keyword"}
        }
    }
}

# RRF 管道
{
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {
                    "technique": "rrf",
                    "parameters": {"rank_constant": 60}
                }
            }
        }
    ]
}
```

---

## API 端点

### 1. 健康检查
```bash
GET /api/v1/health
```

**响应**:
```json
{
  "status": "ok",
  "services": {
    "opensearch": {"status": "healthy", "message": "..."},
    "ollama": {"status": "healthy", "message": "..."}
  }
}
```

### 2. 混合搜索
```bash
POST /api/v1/hybrid-search/
Content-Type: application/json

{
  "query": "Transformer 注意力机制",
  "size": 10,
  "use_hybrid": true
}
```

**响应**:
```json
{
  "query": "Transformer 注意力机制",
  "total": 10,
  "hits": [
    {
      "doc_id": "attention-is-all-you-need",
      "title": "Attention Is All You Need",
      "chunk_text": "The dominant sequence transduction models...",
      "score": 0.95,
      "section_name": "Abstract"
    }
  ],
  "search_mode": "hybrid"
}
```

### 3. Agentic 问答
```bash
POST /api/v1/ask-agentic
Content-Type: application/json

{
  "query": "自注意力机制是如何工作的？",
  "model": "qwen2.5:7b",
  "top_k": 3,
  "use_hybrid": true
}
```

**响应**:
```json
{
  "query": "自注意力机制是如何工作的？",
  "answer": "自注意力机制通过计算查询、键、值矩阵...",
  "sources": ["attention-is-all-you-need"],
  "chunks_used": 3,
  "search_mode": "hybrid",
  "reasoning_steps": [
    "验证查询范围 (score: 85/100)",
    "检索相关文档 (3 chunks)",
    "评估文档相关性 (3/3 relevant)",
    "基于上下文生成回答"
  ],
  "retrieval_attempts": 1,
  "trace_id": "abc123-def456"
}
```

### 4. 提交反馈
```bash
POST /api/v1/feedback
Content-Type: application/json

{
  "trace_id": "abc123-def456",
  "score": 1.0,
  "comment": "回答非常准确"
}
```

---

## 配置说明

### 环境变量 (.env)

```env
# 应用配置
APP_VERSION=0.1.0
DEBUG=true
ENVIRONMENT=development

# OpenSearch
OPENSEARCH__HOST=http://localhost:9200
OPENSEARCH__INDEX_NAME=ai-docs

# Ollama
OLLAMA__HOST=http://localhost:11434
OLLAMA__MODEL=qwen2.5:7b
OLLAMA__TIMEOUT=300

# Embedding (百炼 / Ollama)
EMBEDDING__PROVIDER=bailian
EMBEDDING__BAILIAN_API_KEY=your-bailian-api-key
EMBEDDING__BAILIAN_MODEL=text-embedding-v3
EMBEDDING__BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING__OLLAMA_MODEL=qwen3-embedding
EMBEDDING__DIMENSIONS=1024

# Redis
REDIS__HOST=localhost
REDIS__PORT=6379
REDIS__TTL_HOURS=6

# Langfuse (可选)
LANGFUSE__ENABLED=false
LANGFUSE__PUBLIC_KEY=
LANGFUSE__SECRET_KEY=
```

### 配置层级

使用 `__` 分隔符支持嵌套配置：
- `OPENSEARCH__HOST` → `settings.opensearch.host`
- `EMBEDDING__BAILIAN_API_KEY` → `settings.embedding.bailian_api_key`

---

## 依赖注入

### FastAPI 依赖注入模式

```python
# 定义依赖
def get_settings() -> Settings:
    return Settings()

def get_opensearch_client(settings: Settings = Depends(get_settings)):
    return OpenSearchClient(settings)

# 使用依赖
@router.get("/health")
async def health(
    settings: SettingsDep,  # SettingsDep = Annotated[Settings, Depends(get_settings)]
    opensearch: OpenSearchDep
):
    ...
```

### 优势

- **解耦**: 业务逻辑与依赖创建分离
- **可测试**: 轻松替换 mock 依赖
- **单例**: `lru_cache` 确保依赖只创建一次
- **类型安全**: 完整的类型提示

---

## LangGraph 实现

### 状态定义

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str
    rewritten_query: Optional[str]
    retrieval_attempts: int
    guardrail_result: Optional[GuardrailScoring]
    routing_decision: Optional[RoutingDecision]
    sources: List[SourceItem]
    relevant_sources: List[SourceItem]
    grading_results: List[GradingResult]
```

### 上下文注入

```python
@dataclass
class Context:
    """运行时依赖（不可变）"""
    ollama_client: OllamaClient
    opensearch_client: OpenSearchClient
    embeddings_client: BaseEmbeddingsClient
    langfuse_tracer: Optional[LangfuseTracer]
    model_name: str = "qwen2.5:7b"
    temperature: float = 0.0
    top_k: int = 3

# 创建图
graph = StateGraph(AgentState, context_schema=Context)
```

### 节点注册

```python
graph.add_node("guardrail", ainvoke_guardrail_step)
graph.add_node("retrieve", ToolNode([retrieve_documents]))
graph.add_node("grade", ainvoke_grade_documents_step)
graph.add_node("rewrite", ainvoke_rewrite_query_step)
graph.add_node("generate", ainvoke_generate_answer_step)
graph.add_node("out_of_scope", ainvoke_out_of_scope_step)
```

### 边路由

```python
# 条件路由
graph.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "out_of_scope": "out_of_scope"
    }
)
```

---

## 文档分块策略

### 分块参数

```python
chunk_size = 600      # 目标块大小（token）
overlap_size = 100    # 重叠大小
min_chunk_size = 100  # 最小块大小
section_based = True  # 优先按章节分块
```

### 分块流程

```
原始文档
   ↓
┌────────────────┐
│ 1. 解析章节    │ ← 基于标题（#, ##, ###）
└────────┬───────┘
       ↓
┌────────────────┐
│ 2. 章节分块    │ ← 每个章节作为一个块
└────────┬───────┘
       ↓
   ┌───┴───┐
   │ > 600? │
   └───┬───┘
    是 │   否
    ↓  │    ↓
┌──────────┐  │
│ 滑动窗口 │  │
│ 分块     │  │
└──────────┘  │
       ↓      ↓
    合并所有块
       ↓
┌────────────────┐
│ 3. 生成 Embedding │
└────────────────┘
```

---

## 性能优化

### 1. 缓存策略

```python
# Redis 缓存查询结果
cache_key = f"query:{hash(query)}"
cached_result = await cache.get(cache_key)
if cached_result:
    return cached_result

# 缓存 TTL: 6 小时
await cache.set(cache_key, result, ttl=6 * 3600)
```

### 2. 批量处理

```python
# 批量生成 Embedding
embeddings = await embeddings_client.batch_embed(texts)
```

### 3. 异步处理

```python
# 所有 I/O 操作都是异步
async def retrieve_documents(query: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(...)
```

---

## 部署指南

### Docker Compose

```bash
# 启动所有服务
docker compose up -d

# 查看日志
docker compose logs -f api

# 停止服务
docker compose down
```

### 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| API | 8000 | FastAPI 应用 |
| OpenSearch | 9200 | 向量数据库 |
| OpenSearch Dashboards | 5601 | 可视化管理 |
| Ollama | 11434 | LLM 服务 |
| Redis | 6379 | 缓存 |

### 导入文档

```python
# 示例：导入文档
from src.services.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()
await pipeline.ingest_document(
    file_path="path/to/document.pdf",
    metadata={"source": "arxiv", "category": "cs.AI"}
)
```

---

## 监控与调试

### Langfuse 追踪

```python
# 启用 Langfuse
LANGFUSE__ENABLED=true
LANGFUSE__PUBLIC_KEY=pk-...
LANGFUSE__SECRET_KEY=sk-...

# 访问 Langfuse UI
http://localhost:3001
```

### 查看追踪数据

每个请求都会生成一个 trace，包含：
- Guardrail 评分
- 检索的文档
- 文档评分
- 查询重写（如果有）
- 生成的回答
- 耗时统计

---

## 常见问题

### Q1: Guardrail 评分过低怎么办？

**解决**:
- 降低阈值：`settings.guardrail_threshold = 50`
- 或修改 Guardrail prompt，放宽限制

### Q2: 检索结果不相关？

**解决**:
- 增加 `top_k`：检索更多文档
- 调整混合搜索权重
- 优化 Embedding 模型

### Q3: 回答质量差？

**解决**:
- 检查检索的文档是否相关
- 调整 LLM temperature（降低随机性）
- 优化 Generate prompt

### Q4: 如何更换 LLM 模型？

**解决**:
```bash
# 1. 拉取新模型
ollama pull llama3.2:3b

# 2. 修改配置
OLLAMA__MODEL=llama3.2:3b

# 3. 重启服务
docker compose restart api
```

---

## 开发指南

### 项目结构

```
ai-docs-rag/
├── src/
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── dependencies.py      # 依赖注入
│   ├── routers/             # API 路由
│   │   ├── ping.py         # 健康检查
│   │   ├── search.py       # 搜索端点
│   │   └── agentic_ask.py  # Agentic 问答
│   ├── schemas/             # Pydantic 模型
│   │   └── api/
│   ├── services/            # 业务逻辑
│   │   ├── agents/         # LangGraph Agent
│   │   │   ├── nodes/      # 6 个节点
│   │   │   ├── state.py    # 状态定义
│   │   │   ├── context.py  # 上下文
│   │   │   └── agentic_rag.py
│   │   ├── opensearch/     # OpenSearch 客户端
│   │   ├── ollama/         # Ollama 客户端
│   │   ├── embeddings/     # Embedding 客户端
│   │   ├── cache/          # Redis 缓存
│   │   └── langfuse/       # Langfuse 追踪
│   └── exceptions.py        # 自定义异常
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

### 添加新节点

```python
# 1. 创建节点函数
async def ainvoke_new_step(state: AgentState, runtime: Runtime[Context]):
    # 节点逻辑
    return {"new_field": result}

# 2. 注册节点
graph.add_node("new_node", ainvoke_new_step)

# 3. 添加边
graph.add_edge("previous_node", "new_node")
graph.add_edge("new_node", "next_node")
```

### 测试

```bash
# 运行测试
pytest tests/

# 测试单个端点
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是 Transformer？"}'
```

---

## 总结

AI Docs RAG 展示了如何构建一个生产级的 Agentic RAG 系统，关键特性包括：

1. **智能体工作流**: 6 节点 LangGraph 实现自主决策
2. **混合搜索**: BM25 + 向量搜索，RRF 融合
3. **质量保障**: Guardrail + 文档评分 + 查询重写
4. **可观测性**: Langfuse 全链路追踪
5. **生产就绪**: Docker 部署，缓存，异步处理

这个项目可以作为构建其他 Agentic 应用的参考架构。
