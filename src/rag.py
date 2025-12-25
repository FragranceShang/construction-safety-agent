from graph.state.context import build_context
from graph.state.answer import call_llm, get_llm
from utils.logger import setup_logger
from src.graph.prompt.prompt import REGULATION_QA_PROMPT
from langchain_core.retrievers import BaseRetriever

logger = setup_logger("rag")
def answer_regulation(
    question: str,
    retriever: BaseRetriever,
) -> str:
    """
    Args:
        question (str):
            用户提出的自然语言问题。
        retriever (Any):
            向量检索器，用于检索相关法规条文。

    Returns:
        str:
            基于法规条文生成的回答结果。
    """
    logger.info("Receive question: %s", question)
    # 1. 检索相关法规
    docs = retriever.invoke(question)
    if not docs:
        logger.warning("No relevant regulations found")
        return "未在现有规范中检索到与问题相关的条文。"
    logger.info("Retrieved %d regulation documents: %s", len(docs), [doc.metadata.get("regulation_id") for doc in docs])

    # 2. 构建上下文
    context = build_context(docs)

    # 3. 构建 Prompt
    prompt = REGULATION_QA_PROMPT.format(
        context=context,
        question=question,
    )

    # 4. 调用 LLM
    client = get_llm()
    answer = call_llm(client, prompt)

    logger.info("LLM response generated")
    return answer

# 测试代码
