from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class RegulationState(TypedDict):
    """
    Docstring for RegulationState
    """
    question: str

    # RAG 中间态
    retrieved_docs: List[Document]
    context: str

    # 最终结果
    answer: Optional[str]

    # 控制信息（课程加分点）
    need_answer: bool
