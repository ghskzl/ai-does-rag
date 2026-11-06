import logging
import time
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph

from src.services.embeddings.base import BaseEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    continue_after_guardrail,
)
from .state import AgentState

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """Agentic RAG service with LangGraph workflow.

    This implementation uses:
    - context_schema for dependency injection
    - Runtime[Context] for type-safe access in nodes
    - Direct client invocation (no pre-built runnables)
    - Lightweight nodes as pure functions
    """

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        ollama_client: OllamaClient,
        embeddings_client: BaseEmbeddingsClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        """Initialize agentic RAG service.

        :param opensearch_client: Client for document search
        :param ollama_client: Client for LLM generation
        :param embeddings_client: Client for embeddings
        :param langfuse_tracer: Optional Langfuse tracer
        :param graph_config: Configuration for graph execution
        """
        self.opensearch = opensearch_client
        self.ollama = ollama_client
        self.embeddings = embeddings_client
        self.langfuse_tracer = langfuse_tracer
        self.graph_config = graph_config or GraphConfig()

        logger.info("Initializing AgenticRAGService with configuration:")
        logger.info(f"  Model: {self.graph_config.model}")
        logger.info(f"  Top-k: {self.graph_config.top_k}")
        logger.info(f"  Hybrid search: {self.graph_config.use_hybrid}")
        logger.info(f"  Max retrieval attempts: {self.graph_config.max_retrieval_attempts}")
        logger.info(f"  Guardrail threshold: {self.graph_config.guardrail_threshold}")

        self.graph = self._build_graph()
        logger.info("AgenticRAGService initialized successfully")

    def _build_graph(self):
        """Build and compile the LangGraph workflow.

        Uses context_schema for type-safe dependency injection.
        Nodes are lightweight functions that receive Runtime[Context].

        :returns: Compiled graph ready for invocation
        """
        logger.info("Building LangGraph workflow with context_schema")

        workflow = StateGraph(AgentState, context_schema=Context)

        logger.info("Adding nodes to workflow graph")
        workflow.add_node("guardrail", ainvoke_guardrail_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
        workflow.add_node("retrieve", ainvoke_retrieve_step)
        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)

        logger.info("Configuring graph edges and routing logic")

        workflow.add_edge(START, "guardrail")

        workflow.add_conditional_edges(
            "guardrail",
            continue_after_guardrail,
            {
                "continue": "retrieve",
                "out_of_scope": "out_of_scope",
            },
        )

        workflow.add_edge("out_of_scope", END)

        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("routing_decision", "generate_answer"),
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
            },
        )

        workflow.add_edge("rewrite_query", "retrieve")

        workflow.add_edge("generate_answer", END)

        logger.info("Compiling LangGraph workflow")
        compiled_graph = workflow.compile()
        logger.info("Graph compilation successful")

        return compiled_graph

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
    ) -> dict:
        """Ask a question using agentic RAG.

        :param query: User question
        :param user_id: User identifier for tracing
        :param model: Optional model override
        :returns: Dictionary with answer, sources, reasoning steps, and metadata
        :raises ValueError: If query is empty
        """
        model_to_use = model or self.graph_config.model

        logger.info("=" * 80)
        logger.info("Starting Agentic RAG Request")
        logger.info(f"Query: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Model: {model_to_use}")
        logger.info("=" * 80)

        if not query or len(query.strip()) == 0:
            logger.error("Empty query received")
            raise ValueError("Query cannot be empty")

        trace = None
        if self.langfuse_tracer and self.langfuse_tracer.client:
            logger.info("Creating Langfuse trace")
            metadata = {
                "env": self.graph_config.settings.environment,
                "service": "agentic_rag",
                "top_k": self.graph_config.top_k,
                "use_hybrid": self.graph_config.use_hybrid,
                "model": model_to_use,
            }
            trace = self.langfuse_tracer.client.start_as_current_observation(
                name="agentic_rag_request",
            )

        async def _execute_with_trace():
            """Execute the workflow with or without tracing context."""
            if trace is not None:
                with trace as trace_obj:
                    trace_obj.update(
                        input={"query": query},
                        metadata=metadata,
                        user_id=user_id,
                        session_id=f"session_{user_id}",
                    )
                    logger.debug(f"Trace created: {trace_obj}")
                    return await self._run_workflow(query, model_to_use, user_id, trace_obj)
            else:
                return await self._run_workflow(query, model_to_use, user_id, None)

        try:
            return await _execute_with_trace()
        except Exception as e:
            logger.error(f"Error in Agentic RAG execution: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def _run_workflow(self, query: str, model_to_use: str, user_id: str, trace) -> dict:
        """Execute the workflow with the given trace context."""
        try:
            start_time = time.time()

            logger.info("Invoking LangGraph workflow")

            state_input = {
                "messages": [HumanMessage(content=query)],
                "retrieval_attempts": 0,
                "guardrail_result": None,
                "routing_decision": None,
                "sources": None,
                "relevant_sources": [],
                "relevant_tool_artefacts": None,
                "grading_results": [],
                "metadata": {},
                "original_query": None,
                "rewritten_query": None,
            }

            runtime_context = Context(
                ollama_client=self.ollama,
                opensearch_client=self.opensearch,
                embeddings_client=self.embeddings,
                langfuse_tracer=self.langfuse_tracer,
                trace=trace,
                langfuse_enabled=self.langfuse_tracer is not None and self.langfuse_tracer.client is not None,
                model_name=model_to_use,
                temperature=self.graph_config.temperature,
                top_k=self.graph_config.top_k,
                max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
                guardrail_threshold=self.graph_config.guardrail_threshold,
            )

            config = {"thread_id": f"user_{user_id}_session_{int(time.time())}"}

            if self.langfuse_tracer and trace:
                try:
                    callback_handler = CallbackHandler()
                    config["callbacks"] = [callback_handler]
                    logger.info("CallbackHandler added (will auto-link to current trace)")
                except Exception as e:
                    logger.warning(f"Failed to create CallbackHandler: {e}")

            result = await self.graph.ainvoke(
                state_input,
                config=config,
                context=runtime_context,
            )

            execution_time = time.time() - start_time
            logger.info(f"Graph execution completed in {execution_time:.2f}s")

            answer = self._extract_answer(result)
            sources = self._extract_sources(result)
            retrieval_attempts = result.get("retrieval_attempts", 0)
            reasoning_steps = self._extract_reasoning_steps(result)

            if trace:
                trace.update(
                    output={
                        "answer": answer,
                        "sources_count": len(sources),
                        "retrieval_attempts": retrieval_attempts,
                        "reasoning_steps": reasoning_steps,
                        "execution_time": execution_time,
                    }
                )
                trace.end()
                self.langfuse_tracer.flush()

            logger.info("=" * 80)
            logger.info("Agentic RAG Request Completed Successfully")
            logger.info(f"Answer length: {len(answer)} characters")
            logger.info(f"Sources found: {len(sources)}")
            logger.info(f"Retrieval attempts: {retrieval_attempts}")
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info("=" * 80)

            trace_id = None
            if trace:
                try:
                    trace_id = getattr(trace, "trace_id", None) or self.langfuse_tracer.get_trace_id()
                except Exception:
                    pass

            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "reasoning_steps": reasoning_steps,
                "retrieval_attempts": retrieval_attempts,
                "rewritten_query": result.get("rewritten_query"),
                "execution_time": execution_time,
                "guardrail_score": result.get("guardrail_result").score if result.get("guardrail_result") else None,
                "trace_id": trace_id,
            }

        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            logger.exception("Full traceback:")

            if trace:
                trace.update(output={"error": str(e)}, level="ERROR")
                trace.end()
                self.langfuse_tracer.flush()

            raise

    def _extract_answer(self, result: dict) -> str:
        """Extract final answer from graph result."""
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."

        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    def _extract_sources(self, result: dict) -> List[dict]:
        """Extract sources from graph result, deduplicated by doc_id."""
        sources = []
        seen_doc_ids = set()
        relevant_sources = result.get("relevant_sources", [])

        for source in relevant_sources:
            if hasattr(source, "to_dict"):
                source_dict = source.to_dict()
            elif isinstance(source, dict):
                source_dict = source
            else:
                continue

            doc_id = source_dict.get("doc_id")
            if doc_id and doc_id not in seen_doc_ids:
                sources.append(source_dict)
                seen_doc_ids.add(doc_id)
            elif not doc_id:
                sources.append(source_dict)

        return sources

    def _extract_reasoning_steps(self, result: dict) -> List[str]:
        """Extract reasoning steps from graph result."""
        steps = []
        retrieval_attempts = result.get("retrieval_attempts", 0)
        guardrail_result = result.get("guardrail_result")
        grading_results = result.get("grading_results", [])

        if guardrail_result:
            steps.append(f"Validated query scope (score: {guardrail_result.score}/100)")

        if retrieval_attempts > 0:
            steps.append(f"Retrieved documents ({retrieval_attempts} attempt(s))")

        if grading_results:
            relevant_count = sum(1 for g in grading_results if g.is_relevant)
            steps.append(f"Graded documents ({relevant_count} relevant)")

        if result.get("rewritten_query"):
            steps.append("Rewritten query for better results")

        steps.append("Generated answer from context")

        return steps
