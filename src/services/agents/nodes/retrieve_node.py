import json
import logging
import time
from typing import Dict, Union

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..models import SourceItem
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[int, str, list]]:
    """Retrieve documents directly from OpenSearch and store sources in state.

    Bypasses the ToolNode to preserve document metadata (doc_id, title, source)
    that would otherwise be lost in string serialization.

    :param state: Current agent state
    :param runtime: Runtime context with clients and config
    :returns: Updated state with messages, relevant_sources, retrieval_attempts
    """
    logger.info("NODE: retrieve")
    start_time = time.time()

    messages = state["messages"]
    question = get_latest_query(messages)
    current_attempts = state.get("retrieval_attempts", 0)

    max_attempts = runtime.context.max_retrieval_attempts
    top_k = runtime.context.top_k

    updates = {}
    if state.get("original_query") is None:
        updates["original_query"] = question
        logger.debug(f"Stored original query: {question[:100]}...")

    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_retrieval",
                input_data={
                    "query": question,
                    "attempt": current_attempts + 1,
                    "max_attempts": max_attempts,
                },
                metadata={
                    "node": "retrieve",
                    "top_k": top_k,
                },
            )
            logger.debug(f"Created Langfuse span for retrieval attempt {current_attempts + 1}")
        except Exception as e:
            logger.warning(f"Failed to create span for retrieve node: {e}")

    if current_attempts >= max_attempts:
        logger.warning(f"Max retrieval attempts ({max_attempts}) reached")
        fallback_msg = (
            f"很抱歉，经过 {max_attempts} 次尝试，我未能找到相关的技术文档。\n"
            "这可能是因为：\n"
            "1. 数据库中没有包含相关信息的文档\n"
            "2. 查询术语与索引内容不匹配\n\n"
            "请尝试用更具体的技术术语重新表述您的问题。"
        )

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"status": "max_attempts_reached", "fallback": True},
                metadata={"execution_time_ms": execution_time},
            )

        return {**updates, "messages": [AIMessage(content=fallback_msg)]}

    new_attempt_count = current_attempts + 1
    updates["retrieval_attempts"] = new_attempt_count
    logger.info(f"Retrieval attempt {new_attempt_count}/{max_attempts}")

    try:
        query_embedding = await runtime.context.embeddings_client.embed_query(question)
        logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

        search_results = runtime.context.opensearch_client.search_unified(
            query=question,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=True,
        )

        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        source_items = []
        doc_contents = []

        for i, hit in enumerate(hits):
            source_items.append(
                SourceItem(
                    doc_id=hit["doc_id"],
                    title=hit.get("title", ""),
                    source=hit.get("source", ""),
                    relevance_score=hit.get("score", 0.0),
                )
            )

            chunk_text = hit["chunk_text"]
            section = hit.get("section_name", "")
            title = hit.get("title", "") or "未知标题"
            doc_id = hit["doc_id"]
            formatted = f"[{title}] (章节: {section})\n{chunk_text}"
            doc_contents.append(formatted)

        combined_context = "\n\n---\n\n".join(doc_contents) if doc_contents else "No documents found."

        tool_call_id = f"retrieve_{new_attempt_count}"

        updates["relevant_sources"] = source_items
        updates["messages"] = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "retrieve_documents",
                        "args": {"query": question},
                    }
                ],
            ),
            ToolMessage(
                content=combined_context,
                tool_call_id=tool_call_id,
                name="retrieve_documents",
            ),
        ]

        logger.info(f"Stored {len(source_items)} source items in state")

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        error_context = f"Retrieval error: {str(e)}"
        tool_call_id = f"retrieve_{new_attempt_count}"

        updates["relevant_sources"] = []
        updates["messages"] = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "retrieve_documents",
                        "args": {"query": question},
                    }
                ],
            ),
            ToolMessage(
                content=error_context,
                tool_call_id=tool_call_id,
                name="retrieve_documents",
            ),
        ]

    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "status": "completed",
                "documents_retrieved": len(updates.get("relevant_sources", [])),
                "attempt": new_attempt_count,
            },
            metadata={"execution_time_ms": execution_time},
        )

    return updates
