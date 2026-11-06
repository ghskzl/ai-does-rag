import logging
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_out_of_scope_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Handle out-of-scope queries with a helpful message.

    This node responds to queries that are outside the domain of
    AI technical documents with a polite, informative message.

    :param state: Current agent state
    :param runtime: Runtime context (not used in this node)
    :returns: Dictionary with messages containing the out-of-scope response
    """
    logger.info("NODE: out_of_scope")

    question = get_latest_query(state["messages"])

    response_text = (
        "很抱歉，我只能帮助回答关于 AI 技术文档的问题，"
        "包括人工智能、机器学习和深度学习相关的技术研究。\n\n"
        f"您的问题：'{question}'\n\n"
        "这似乎超出了我的专业领域。对于这类问题，您可以尝试：\n"
        "- 通用 AI 助手来获取广泛的知识问答\n"
        "- 专业领域资源来处理 AI/ML 之外的主题\n"
        "- 技术文档来查询特定软件/工具的使用方法\n\n"
        "如果您有关于 AI 技术文档的问题，我很乐意帮助！"
    )

    logger.info("Responding with out-of-scope message")

    return {"messages": [AIMessage(content=response_text)]}
