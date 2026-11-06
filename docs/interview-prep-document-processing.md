# 文档处理模块面试准备

## 一、模块架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      文档入库管道 (Ingestion Pipeline)            │
│                                                                 │
│  PDF/MD/TXT ──→ Parser ──→ Chunker ──→ Embedding ──→ OpenSearch│
│                 │            │            │              │       │
│           RawChunk      TextChunk    List[float]     bulk index  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心文件清单

| 文件 | 职责 | 关键类/函数 |
|------|------|------------|
| `document_parser.py` | 文档解析 | `PDFDocumentParser`, `MarkdownDocumentParser`, `PlainTextParser`, `RawChunk` |
| `chunker.py` | 文本分块 | `TextChunker`, `TextChunk` |
| `ingestion_service.py` | 管道编排 | `DocumentIngestionService`, `IngestedDocument` |
| `ollama_embeddings_client.py` | 向量化 | `OllamaEmbeddingsClient` |
| `opensearch/client.py` | 索引与搜索 | `OpenSearchClient` |
| `opensearch/index_config.py` | 索引定义 | `DOCS_CHUNKS_MAPPING`, `HYBRID_RRF_PIPELINE` |
| `opensearch/query_builder.py` | 查询构建 | `QueryBuilder` |
| `config.py` | 配置管理 | `ChunkingSettings`, `OpenSearchSettings`, `EmbeddingSettings` |

---

## 二、核心内容详解

### 2.1 文档解析层 — `document_parser.py`

#### 设计模式：策略模式 (Strategy Pattern)

```python
class BaseDocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> List[RawChunk]: ...

def get_document_parser(file_path: Path) -> BaseDocumentParser:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":       return PDFDocumentParser()
    if suffix in (".md", ...): return MarkdownDocumentParser()
    if suffix in (".txt", ...):return PlainTextParser()
    raise ValueError(f"Unsupported file type: {suffix}")
```

**面试要点**：
- 使用抽象基类 (ABC) 定义统一接口
- 工厂函数 `get_document_parser()` 根据文件后缀选择策略
- 新增文件类型只需添加新 Parser 类 + 工厂分支，符合开闭原则 (OCP)

#### PDF 解析 — PyMuPDF

```python
class PDFDocumentParser(BaseDocumentParser):
    def parse(self, file_path: Path) -> List[RawChunk]:
        doc = pymupdf.open(str(file_path))
        try:
            chunks: List[RawChunk] = []
            current_offset = 0
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if not text or not text.strip():
                    current_offset += len(text)
                    continue
                chunks.append(RawChunk(
                    text=text.strip(),
                    page_number=page_num + 1,
                    section_title=f"Page {page_num + 1}",
                    start_char=current_offset,
                    end_char=current_offset + len(text),
                ))
                current_offset += len(text)
            return chunks
        finally:
            doc.close()
```

**关键设计决策**：

1. **逐页提取**：每页生成一个 `RawChunk`，保留页码信息
   - 优点：保留文档结构，可追溯到具体页面
   - 缺点：跨页内容可能被截断（后续由 chunker 处理）

2. **字符偏移量追踪**：`start_char` / `end_char` 累积计算
   - 用途：支持原文定位，用户点击引用可跳转到 PDF 具体位置

3. **资源管理**：`try/finally` 确保 `doc.close()`
   - PyMuPDF 打开的文件句柄必须显式关闭，否则在大量文件处理时会 FD 泄漏

4. **空白页跳过**：`if not text or not text.strip()` 但依然更新 offset
   - 保持偏移量的正确性

#### RawChunk 使用 `__slots__` 优化

```python
class RawChunk:
    __slots__ = ("text", "page_number", "section_title", "start_char", "end_char")
```

**面试要点**：
- `__slots__` 替代 `__dict__`，每个实例节省约 40-50% 内存
- 对于大量 chunk（如 90+ 个），内存优化效果显著
- 代价：不能动态添加属性

---

### 2.2 文本分块层 — `chunker.py`

#### 两种分块策略对比

| 特性 | Section-Based（默认） | Flat |
|------|----------------------|------|
| 输入粒度 | 每个 RawChunk 独立处理 | 所有文本拼接后统一处理 |
| 结构感知 | 保留页面/章节边界 | 无结构感知 |
| 跨页合并 | 不会跨页 | 可能跨页 |
| 适用场景 | 结构化文档（PDF论文章节） | 纯文本、日志 |
| chunk_index | 全局递增 | 按顺序编号 |

#### Section-Based Chunking 核心逻辑

```python
def _section_based_chunk(self, raw_chunks: List[RawChunk]) -> List[TextChunk]:
    for raw_chunk in raw_chunks:
        text = raw_chunk.text.strip()
        if not text:
            continue

        if len(text) <= self.chunk_size:          # 600 字符
            if len(text) >= self.min_chunk_size:   # 100 字符
                → 直接作为一个 TextChunk
            else:
                → 丢弃（太短的块是噪声）
            continue

        # 超过 chunk_size，需要进一步分割
        sub_chunks = self._split_text(text, raw_chunk.start_char)
        for sub_text, start, end in sub_chunks:
            if len(sub_text) >= self.min_chunk_size:
                → 创建 TextChunk
```

**关键参数**（来自 `ChunkingSettings`）：
- `chunk_size = 600` — 目标块大小（字符数）
- `overlap_size = 100` — 相邻块重叠字符数
- `min_chunk_size = 100` — 最小块大小阈值
- `section_based = True` — 默认启用结构化分块

#### `_split_text()` — 核心分割算法

```python
def _split_text(self, text: str, base_offset: int) -> List[tuple]:
    results: List[tuple] = []
    start = 0
    while start < len(text):
        end = min(start + self.chunk_size, len(text))

        # 关键：优先在换行符处断开
        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + self.chunk_size // 2:  # > 300
                end = newline_pos

        chunk_text = text[start:end].strip()
        if chunk_text:
            results.append((chunk_text, base_offset + start, base_offset + end))

        # 重叠处理
        start = end - self.overlap_size if end < len(text) else end
    return results
```

**算法图解**（chunk_size=600, overlap=100）：

```
文本: |──────── 2000 字符 ────────|
第1块: |──600──|
第2块:      |──500──600──|    ← 从 500 开始（600-100=500）
第3块:            |──500──600──|
...
```

**面试重点**：

1. **换行符优先**：`rfind("\n", start, end)` 从后往前找换行符
   - 条件 `newline_pos > start + chunk_size // 2` 确保断点在后半段
   - 避免在段落中间断开，保持语义完整性

2. **重叠机制**：`start = end - overlap_size`
   - 为什么需要？避免关键上下文恰好在块边界被切断
   - 100 字符 ≈ 2-3 句话，足以保留上下文

3. **偏移量正确性**：`base_offset + start` 确保全局定位

#### 为什么选择 600 字符？

**面试回答思路**：
- Embedding 模型的输入限制通常在 512 tokens（约 400-600 英文单词 / 200-400 中文字）
- 600 字符 ≈ 150-300 tokens，在模型窗口内
- 太小：上下文不足，检索质量差
- 太大：超出 embedding 模型窗口，信息被截断
- 600 是经验值，可根据实际模型和文档类型调优

---

### 2.3 向量化层 — `OllamaEmbeddingsClient`

```python
class OllamaEmbeddingsClient(BaseEmbeddingsClient):
    async def embed_passages(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self._embed(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": texts}
        response = await self.client.post(f"{self.base_url}/api/embed", json=payload)
        response.raise_for_status()
        return response.json()["embeddings"]
```

**设计要点**：

1. **批量处理**：`batch_size=50`
   - 避免单次请求过大导致 OOM 或超时
   - Ollama 在 CPU 模式下 embedding 较慢，批量可提升吞吐

2. **超时配置**：`httpx.AsyncClient(timeout=120.0)`
   - Embedding 可能耗时较长（尤其 CPU 模式），120s 超时

3. **接口抽象**：`BaseEmbeddingsClient` (ABC)
   - 支持多 provider 切换（Ollama / 百炼）
   - 策略模式，运行时可替换

4. **异步设计**：全部 `async/await`
   - FastAPI 是异步框架，embedding 调用不阻塞主线程

---

### 2.4 索引层 — OpenSearch

#### 索引结构（`index_config.py`）

```python
DOCS_CHUNKS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "index.knn.space_type": "cosinesimil",
        "analysis": {
            "analyzer": {
                "text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"],
                }
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {"ef_construction": 512, "m": 16},
                },
            },
            # ... 其他字段
        },
    },
}
```

**面试要点**：

1. **`dynamic: "strict"`**：严格模式，未定义字段写入会报错
   - 防止脏数据污染索引

2. **双字段设计**（`chunk_text` + `chunk_text.keyword`）：
   - `text` 类型：全文检索（BM25），经过 analyzer 分词
   - `keyword` 类型：精确匹配、聚合、排序

3. **HNSW 向量索引**：
   - `ef_construction=512`：构建时搜索范围，越大越精确但越慢
   - `m=16`：每个节点的最大连接数，影响查询速度和内存
   - `cosinesimil`：余弦相似度，适合文本 embedding

4. **自定义 Analyzer**：
   - `standard` 分词器 → `lowercase` → `stop`（去停用词）→ `snowball`（词干提取）
   - 适合英文技术文档

#### 混合搜索 — BM25 + KNN + RRF

```python
def _search_hybrid(self, query, query_embedding, size, min_score):
    # 1. 构建 BM25 查询
    bm25_body = QueryBuilder(query=query, size=size * 2).build()
    bm25_query = bm25_body["query"]

    # 2. 构建混合查询
    hybrid_query = {
        "hybrid": {
            "queries": [
                bm25_query,                                              # BM25
                {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}}  # KNN
            ]
        }
    }

    # 3. 通过 RRF pipeline 融合
    response = self.client.search(
        index=self.index_name,
        body={"size": size, "query": hybrid_query, ...},
        params={"search_pipeline": "hybrid-rrf-pipeline"}
    )
```

**RRF (Reciprocal Rank Fusion) 公式**：

```
RRF_score(d) = Σ 1 / (rank_constant + rank_i(d))
```

- `rank_constant = 60`（默认值）
- 每路检索结果按排名贡献分数，与绝对分数无关
- 优点：消除不同检索方式分数尺度差异

**为什么要混合搜索？**

| 检索方式 | 擅长 | 不擅长 |
|---------|------|--------|
| BM25（关键词） | 精确匹配专有名词、代码 | 同义词、语义理解 |
| KNN（向量） | 语义相似度 | 精确关键词匹配 |
| RRF 融合 | 两者兼顾 | — |

**面试加分点**：
- `size * 2`：每路检索 2x 数量，融合后取 top-N，提高召回率
- RRF 无需调权重，比线性加权更鲁棒

---

### 2.5 管道编排层 — `DocumentIngestionService`

```python
class DocumentIngestionService:
    async def embed_and_index_chunks(self, chunks, doc_id, title, source):
        # 1. 批量向量化
        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embeddings.embed_passages(chunk_texts)

        # 2. 构建 bulk items
        bulk_items = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_data = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.text,
                # ... 其他元数据
                "embedding_model": embedding_model,
                "created_at": now,
            }
            bulk_items.append({"chunk_data": chunk_data, "embedding": embedding})

        # 3. 批量索引
        result = self.opensearch.bulk_index_chunks(bulk_items)
        return result["success"]
```

**设计要点**：

1. **chunk_id 命名规范**：`{doc_id}_chunk_{index}`
   - 全局唯一，支持按 doc_id 前缀过滤
   - 支持幂等更新（重复上传相同 doc_id 会覆盖）

2. **记录 embedding_model**：
   - 支持未来更换模型后，区分不同版本 embedding
   - 避免不同模型生成的向量混合导致相似度计算失效

3. **bulk 索引**：使用 `opensearchpy.helpers.bulk`
   - 批量写入比逐条写入快 10-100x
   - `refresh=True` 确保立即可搜索

---

## 三、数据流全链路

```
用户上传 PDF (FastAPI UploadFile)
    │
    ▼
[1] PDFDocumentParser.parse()
    │  PyMuPDF 逐页提取
    │  15页 PDF → 15 个 RawChunk
    │
    ▼
[2] TextChunker.chunk_document()
    │  Section-based 分块 (chunk_size=600, overlap=100)
    │  15 个 RawChunk → 90 个 TextChunk
    │  (每页可能产生多个 chunk)
    │
    ▼
[3] OllamaEmbeddingsClient.embed_passages()
    │  批量向量化 (batch_size=50)
    │  90 个文本 → 90 个 1024维向量
    │  模型: qwen3-embedding:0.6b
    │
    ▼
[4] OpenSearchClient.bulk_index_chunks()
    │  opensearchpy.helpers.bulk 批量写入
    │  每个 chunk 包含: text + embedding + metadata
    │
    ▼
可搜索状态 ✓
    │
    ▼
[5] 查询时: search_unified()
    │  BM25 + KNN → RRF 融合
    │  返回 top-K 结果 + 高亮片段
```

---

## 四、高频面试问题与回答

### Q1: 为什么选择 section-based chunking 而不是固定长度切分？

**A**: 固定长度切分（如每 500 字一刀）会在段落中间断开，破坏语义完整性。Section-based 策略：
- 尊重文档原有结构（页面、章节）
- 每个 chunk 尽量保持完整语义单元
- 重叠机制（overlap=100）进一步缓解边界截断问题

实测对比：在 "Attention Is All You Need" 论文上，section-based 产生的 90 个 chunk 比 flat chunking 的语义连贯性明显更好。

### Q2: overlap 重叠 100 字符是怎么确定的？

**A**: 这是经验值 + 调优的结果：
- 太小（如 20 字符）：上下文保留不足
- 太大（如 300 字符）：冗余数据多，浪费存储和 embedding 计算
- 100 字符 ≈ 2-3 个英文句子，足以保留过渡上下文
- 实际项目中可通过 A/B 测试不同参数组合，用检索准确率评估

### Q3: chunk_size=600 的依据是什么？和 embedding 模型有什么关系？

**A**: 直接受限于 embedding 模型的输入窗口：
- qwen3-embedding:0.6b 的最大输入约 512 tokens
- 600 英文字符 ≈ 150 tokens，在窗口内有余量
- 如果 chunk 太长超出模型窗口，embedding 会截断，丢失信息
- 中文场景需调小（1 中文字 ≈ 1-2 tokens）

### Q4: 为什么用 HNSW 而不是暴力搜索 (flat)？

**A**: 性能权衡：
- 暴力搜索：100% 召回率，但 O(N) 复杂度，10万向量需逐一比较
- HNSW：O(log N) 查询，通过多层图结构实现近似最近邻
- 本项目参数：`ef_construction=512, m=16`
  - ef_construction 大 → 构建慢但查询精确
  - m=16 → 每节点 16 个连接，平衡内存和精度
- 对于 < 10万文档的规模，HNSW 是最佳选择

### Q5: RRF 融合相比线性加权有什么优势？

**A**: 线性加权 `score = α * bm25_score + (1-α) * knn_score` 的问题：
- BM25 和 KNN 的分数尺度完全不同，难以归一化
- α 需要人工调优，不同查询可能需要不同 α

RRF 的优势：
- 只依赖排名，不依赖绝对分数
- 公式 `1/(k + rank)` 天然归一化
- `k=60` 是经验常数，通常不需要调优
- 对不同检索方式的分数分布差异免疫

### Q6: `__slots__` 的作用和代价？

**A**: 
- 作用：阻止创建 `__dict__`，每个实例节省约 40-50% 内存
- 对于 RawChunk（大量实例），内存优化显著
- 代价：不能动态添加新属性
- 适用场景：数据类、DTO、大量实例的对象

### Q7: 如何处理 PDF 中的表格和图片？

**A**: 当前实现的局限：
- PyMuPDF 的 `get_text("text")` 只提取文本，表格可能丢失结构
- 图片完全被忽略

改进方案（面试扩展回答）：
1. **表格**：使用 `pymupdf` 的 `get_text("blocks")` 或 `find_tables()` API
2. **图片**：OCR（如 Tesseract）或视觉模型（如 GPT-4V）生成描述
3. **公式**：LaTeX 提取工具（如 `pix2tex`）
4. **多模态**：使用 LayoutLM 等文档理解模型

### Q8: 如果两个不同文档的 chunk 内容完全相同怎么办？

**A**: 当前实现：两个 chunk 都会索引（chunk_id 不同，因为包含 doc_id）。
改进方案：
1. 计算文本 hash（SHA-256），去重
2. 使用 OpenSearch 的 `_id` 去重（相同 hash → 相同 _id → 覆盖）
3. 好处：节省存储、避免检索时返回重复内容

### Q9: 批量索引时如果部分失败怎么处理？

**A**: 当前实现：
```python
success, failed = helpers.bulk(self.client, actions, refresh=True)
return {"success": success, "failed": len(failed)}
```
- `helpers.bulk` 返回成功数和失败列表
- 当前只记录数量，未重试

改进方案：
1. 记录失败的 chunk_id，日志输出
2. 指数退避重试（最多 3 次）
3. 部分失败时返回部分成功状态，而非整体失败

### Q10: 搜索时 `size * 2` 是什么意思？

**A**: 混合搜索中，每路（BM25 和 KNN）各检索 `size * 2` 条结果，融合后取 top-`size` 条。
- 原因：RRF 融合后排名可能变化，多检索一些保证最终结果数量充足
- 类似数据库查询中的 "oversampling" 策略
- `hybrid_search_size_multiplier = 2` 是可配置的

---

## 五、设计模式总结

| 模式 | 应用位置 | 说明 |
|------|---------|------|
| **策略模式** | DocumentParser, EmbeddingsClient | 抽象接口 + 多实现，运行时切换 |
| **工厂模式** | `get_document_parser()` | 根据文件类型创建对应 Parser |
| **管道模式** | IngestionService | parse → chunk → embed → index |
| **数据类模式** | RawChunk (`__slots__`), TextChunk (`@dataclass`) | 纯数据载体，结构化传递 |
| **批量处理** | embed_passages, bulk_index | 分批处理大量数据，控制内存和超时 |

---

## 六、关键配置参数速查

```python
# ChunkingSettings
chunk_size: int = 600          # 目标块大小（字符数）
overlap_size: int = 100        # 相邻块重叠字符数
min_chunk_size: int = 100      # 最小块大小（低于则丢弃）
section_based: bool = True     # 启用结构化分块

# OpenSearchSettings
vector_dimension: int = 1024   # 向量维度（匹配 embedding 模型）
vector_space_type: str = "cosinesimil"  # 余弦相似度
rrf_pipeline_name: str = "hybrid-rrf-pipeline"
hybrid_search_size_multiplier: int = 2

# EmbeddingSettings
provider: Literal["bailian", "ollama"] = "bailian"
ollama_model: str = "qwen3-embedding"
dimensions: int = 1024

# HNSW Parameters
ef_construction: 512   # 构建精度
m: 16                  # 节点连接数
```

---

## 七、可以进一步优化的方向

1. **语义分块 (Semantic Chunking)**：用 embedding 相似度检测语义断点，而非固定长度
2. **父子块 (Parent-Child)**：检索时匹配小块，返回时给大块上下文
3. **查询改写 (Query Rewriting)**：用 LLM 扩展用户查询，提高召回率
4. **Reranker**：检索后用 Cross-Encoder 重排序，提高精度
5. **多模态解析**：表格、图片、公式的结构化提取
6. **增量索引**：检测文档变更，只重新索引变化的部分
7. **缓存层**：Redis 缓存热门查询的 embedding 和检索结果
