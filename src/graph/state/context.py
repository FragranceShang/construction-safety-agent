from typing import List
from langchain_core.documents import Document
from model.state import RegulationState

def context_node(state: RegulationState) -> RegulationState:
    state["context"] = build_context(state["retrieved_docs"])
    return state

def build_context(docs: List[Document]) -> str:
    """
    构建供大模型使用的上下文文本。

    将向量检索返回的条例文档列表，按照“条例编号 + 条文内容”的格式
    拼接成一段连续文本，用于作为 LLM Prompt 的上下文输入。

    Args:
        docs (List[Document]):
            向量检索返回的 Document 列表。
            - doc.page_content: 条例正文内容
            - doc.metadata["regulation_id"]: 条例编号（如 "3.1.2"）

    Returns:
        str:
            拼接后的上下文字符串，每条条例之间使用空行分隔。
    """
    parts: List[str] = []
    for doc in docs:
        reg_id = doc.metadata.get("regulation_id", "")
        if reg_id:
            parts.append(f"【{reg_id}】{doc.page_content}")
    return "\n\n".join(parts)
