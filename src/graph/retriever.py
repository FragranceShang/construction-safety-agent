from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from builder import get_embedding
from model.state import RegulationState

def retrieve_node(state: RegulationState) -> RegulationState:
    """
    基于 RAG 框架的法规检索节点。
    Args:
        state (RegulationState):
            包含问题和其他中间状态的字典。
    Returns:
        RegulationState:
            更新后的状态，包含检索到的文档和是否需要回答的标志。
    """
    retriever = build_retriever(load_vectorstore("faiss_index"), top_k=9)
    question = state["question"]

    docs = retriever.invoke(question)

    state["retrieved_docs"] = docs
    state["need_answer"] = len(docs) > 0
    return state

def load_vectorstore(path: str) -> FAISS:
    """
    加载本地 FAISS 向量库
    Args:
        path (str): 向量库保存路径
    Returns:
        FAISS: 加载的 FAISS 向量库
    """
    embedding = get_embedding()

    vectorstore = FAISS.load_local(
        folder_path=path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def build_retriever(
    vectorstore: FAISS,
    top_k: int = 5,
) -> BaseRetriever:
    """
    基于 FAISS 向量库构建法规检索器（Retriever）。

    该函数负责将底层的向量存储（FAISS）封装为 LangChain
    统一的 Retriever 抽象，用于后续的 RAG 检索阶段。

    Args:
        vectorstore (FAISS):
            已加载并初始化完成的 FAISS 向量库实例。
        top_k (int):
            每次检索返回的最相关法规条文数量。

    Returns:
        BaseRetriever:
            符合 LangChain Retriever 接口规范的检索器实例。
    """
    return vectorstore.as_retriever(
        search_kwargs={
            "k": top_k
        }
    )

if __name__ == "__main__":
    vectorstore = load_vectorstore("faiss_index")

    query = "施工现场供用电系统应满足哪些要求？"

    results = vectorstore.similarity_search(query, k=6)

    for doc in results:
        print("----")
        print(doc.page_content)
        print(doc.metadata)
