import logging

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.services.embeddings.base import BaseEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: BaseEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
):
    """Create a retriever tool that wraps OpenSearch service.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing embeddings service
    :param top_k: Number of chunks to retrieve
    :param use_hybrid: Use hybrid search (BM25 + vector)
    :returns: LangChain tool for retrieving documents
    """

    @tool
    async def retrieve_documents(query: str) -> list[Document]:
        """搜索并返回相关的 AI 技术文档。

        当用户询问以下主题时使用此工具：
        - 机器学习概念或技术
        - 深度学习架构
        - 自然语言处理
        - 计算机视觉方法
        - AI 研究主题
        - 特定算法或模型

        :param query: 描述要查找什么文档的搜索查询
        :returns: 包含元数据的相关文档片段列表
        """
        logger.info(f"Retrieving documents for query: {query[:100]}...")
        logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

        logger.debug("Generating query embedding")
        query_embedding = await embeddings_client.embed_query(query)
        logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

        logger.debug("Searching OpenSearch")
        search_results = opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid,
        )

        documents = []
        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        for hit in hits:
            doc = Document(
                page_content=hit["chunk_text"],
                metadata={
                    "doc_id": hit["doc_id"],
                    "title": hit.get("title", ""),
                    "source": hit.get("source", ""),
                    "score": hit.get("score", 0.0),
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                },
            )
            documents.append(doc)

        logger.debug(f"Converted {len(documents)} hits to LangChain Documents")
        logger.info(f"Retrieved {len(documents)} documents successfully")

        return documents

    return retrieve_documents
