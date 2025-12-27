from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class RegulationState(TypedDict):
    """
    Docstring for RegulationState
    """
    question: str

    # RAG 中间态
    memory: any
    retrieved_docs: List[Document]
    history: str
    context: str

    # 最终结果
    answer: Optional[str]

    # 控制信息
    need_answer: bool
