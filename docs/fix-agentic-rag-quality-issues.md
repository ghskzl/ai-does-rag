# Agentic RAG 问答质量修复文档

**日期**: 2026-10-07  
**修复范围**: sources 字段为空、LLM 输出冗长、检索覆盖率不足

---

## 问题概述

在对 Transformer 架构问答进行质量评估后，发现三个核心问题：

| 问题 | 类型 | 严重程度 |
|------|------|----------|
| `sources` 字段返回空数组 | Bug | 高 |
| LLM 输出包含过多自我解释 | Prompt 设计 | 中 |
| Layer Norm 内容未被检索到 | 检索策略 | 低 |

---

## 问题 1：Sources 字段为空

### 现象

API 返回的 `sources` 字段始终为空数组 `[]`，前端无法展示引用来源。

### 根因分析

通过追踪数据流发现完整的断裂链：

```
retrieve_node → ToolNode → grade_documents → generate_answer → _extract_sources
     ↑                                                              ↓
  创建 tool_calls                                          读取 state["relevant_sources"]
  (不存储元数据)                                              (始终为空)
```

1. **`extract_sources_from_tool_messages()` 是空函数**（`nodes/utils.py:11-24`）
   ```python
   def extract_sources_from_tool_messages(messages: List) -> List[SourceItem]:
       sources = []
       for msg in messages:
           if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
               if msg.name == "retrieve_documents":
                   pass  # 什么都没做
       return sources  # 始终返回 []
   ```

2. **ToolNode 丢失元数据**：LangChain 的 `ToolNode` 处理 `Document` 对象时，metadata（doc_id, title, source）在序列化为 `ToolMessage.content` 字符串时丢失。

3. **没有任何节点填充 `state["relevant_sources"]`**：`retrieve_node` 只创建 tool_calls，`grade_documents_node` 和 `generate_answer_node` 都不写入该字段。

### 解决策略

**核心思路**：绕过 ToolNode，在 retrieve_node 中直接调用搜索并存储元数据。

```
修复前: retrieve → tools_condition → ToolNode → grade_documents
修复后: retrieve → grade_documents（直接连接）
```

### 实现细节

**文件 1: `src/services/agents/nodes/retrieve_node.py`**

重写检索逻辑：
- 直接调用 `embeddings_client.embed_query()` 生成查询向量
- 直接调用 `opensearch_client.search_unified()` 执行混合搜索
- 将每个 hit 转换为 `SourceItem` 对象，存入 `state["relevant_sources"]`
- 手动创建 `AIMessage`（含 tool_calls）+ `ToolMessage`（含格式化文档内容）
- `ToolMessage.content` 保留为字符串，确保 `get_latest_context()` 正常工作

```python
# 关键代码片段
source_items = []
doc_contents = []

for i, hit in enumerate(hits):
    source_items.append(SourceItem(
        doc_id=hit["doc_id"],
        title=hit.get("title", ""),
        source=hit.get("source", ""),
        relevance_score=hit.get("score", 0.0),
    ))
    formatted = f"[文档 {i+1}] (ID: {doc_id}, 标题: {title}, 章节: {section})\n{chunk_text}"
    doc_contents.append(formatted)

updates["relevant_sources"] = source_items
updates["messages"] = [
    AIMessage(content="", tool_calls=[...]),
    ToolMessage(content=combined_context, tool_call_id=tool_call_id, name="retrieve_documents"),
]
```

**文件 2: `src/services/agents/agentic_rag.py`**

简化图路由：
- 移除 `ToolNode` 和 `tools_condition` 导入
- 移除 `tool_retrieve` 节点
- 移除 `create_retriever_tool` 导入
- 将 `retrieve → tools_condition → tool_retrieve → grade_documents` 改为 `retrieve → grade_documents`

**文件 3: `src/schemas/api/ask.py`**

修正响应模型：
- `sources: List[str]` → `sources: List[Dict[str, Any]]`
- 匹配实际返回的结构化来源数据

### 验证结果

```
来源数量: 5（修复前: 0）
来源 1: Attention Is All You Need (ID: 20261006155753_1706_03762v7)
        来源: 1706.03762v7.pdf, 相关度: 0.0164
来源 2: Attention Is All You Need (ID: 20261006155753_1706_03762v7)
        来源: 1706.03762v7.pdf, 相关度: 0.0164
...
```

---

## 问题 2：LLM 输出过于冗长

### 现象

qwen3:4b 在回答中添加大量自我解释段落：
- "根据提供的文档..."
- "我遵循了指令..."
- "基于检索结果..."
- "我没有编造信息..."

### 根因分析

`GENERATE_ANSWER_PROMPT` 中的指令触发了小模型的过度合规行为：
- "仅基于检索到的文档提供全面、准确的回答" → 模型认为需要解释自己是如何遵循这个指令的
- "不要编造信息" → 模型添加声明表示自己没有编造
- 缺少直接性指令 → 模型默认添加元描述

这是小模型（4B 参数）的典型行为：对指令的理解过于字面，倾向于过度解释。

### 解决策略

在 prompt 中添加明确的直接性约束，禁止元描述。

### 实现细节

**文件: `src/services/agents/prompts.py`**

在 `GENERATE_ANSWER_PROMPT` 的指令部分新增：
```
- 直接回答问题，不要解释你的回答方式或声明你遵循了什么原则
- 不要添加关于"根据提供的文档"或"基于检索结果"等元描述
```

### 验证结果

修复前：
```
根据提供的文档信息，我将严格基于检索到的技术文档回答您的问题...
以下是基于检索到的文档中对 Transformer 架构组件的详细描述...
我的回答完全基于提供的文档，未添加任何外部信息或编造内容...
```

修复后：
```
Transformer架构的主要组件包括以下内容：
- 编码器堆栈：由6个堆叠的层组成...
- 解码器堆栈：结构与编码器堆栈类似...
- 嵌入层：将输入和输出标记转换为向量...
```

---

## 问题 3：Layer Norm 未被检索

### 现象

询问"Transformer 架构的所有主要组件"时，回答中未提及 Layer Normalization。

### 根因分析

1. **数据库验证**：Layer Norm 内容存在（29 个 chunks 提到 "LayerNorm"）
2. **排序问题**：对于"所有主要组件"这类宽泛查询，Self-Attention 和 Encoder/Decoder 描述的语义相似度高于 Layer Norm 的具体实现细节
3. **top_k 限制**：原 top_k=3 只返回 3 个 chunks，不足以覆盖所有组件

### 解决策略

增加 top_k 从 3 到 5，提高检索覆盖率。

### 实现细节

修改三个文件中的 top_k 默认值：

| 文件 | 修改 |
|------|------|
| `src/services/agents/context.py` | `top_k: int = 3` → `5` |
| `src/services/agents/config.py` | `top_k: int = 3` → `5` |
| `src/services/agents/factory.py` | `top_k: int = 3` → `5` |

### 验证结果

- 检索文档数：3 → 5
- 覆盖率提升，但 Layer Norm 仍未进入 top-5（排序分数低于 Self-Attention 描述）
- 这是混合搜索排序的固有限制，非 bug

### 后续优化方向

如需完整覆盖所有组件，可考虑：
1. 进一步提高 top_k（但会增加 LLM 处理时间）
2. 实现多轮检索（针对不同子查询分别检索）
3. 使用 query decomposition 将宽泛查询拆分为多个具体查询
4. 调整混合搜索的 BM25/向量权重比例

---

## 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `src/services/agents/nodes/retrieve_node.py` | 重写 | 直接调用搜索，存储 SourceItem |
| `src/services/agents/agentic_rag.py` | 重构 | 移除 ToolNode，简化图路由 |
| `src/services/agents/prompts.py` | 优化 | 添加直接性指令 |
| `src/services/agents/context.py` | 配置 | top_k: 3 → 5 |
| `src/services/agents/config.py` | 配置 | top_k: 3 → 5 |
| `src/services/agents/factory.py` | 配置 | top_k: 3 → 5 |
| `src/schemas/api/ask.py` | 修正 | sources 类型: List[str] → List[Dict] |

---

## 性能指标

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 来源数量 | 0 | 5 |
| 执行时间（GPU） | ~88s | ~77-154s |
| 检索次数 | 1 | 1-2（含查询重写） |
| 回答质量 | 冗长、自我解释 | 直接、结构化 |

执行时间增加的原因：top_k 从 3 增加到 5，LLM 需要处理更多上下文。

---

## 架构变更图

```
修复前:
START → guardrail → retrieve → tools_condition → ToolNode → grade_documents → generate_answer → END
                                      ↓
                                     END

修复后:
START → guardrail → retrieve → grade_documents → generate_answer → END
                                ↓
                          rewrite_query → retrieve（循环）
```

关键变化：retrieve 节点从"创建 tool_call 请求"变为"直接执行检索并存储结果"，消除了 ToolNode 中间层，保留了元数据完整性。
