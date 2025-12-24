from langchain_core.documents import Document
from typing import List
from model.regulation import Regulation
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_faiss_vectorstore(documents: List[Document], save_path: str) -> FAISS:
    """
    构建 FAISS 向量库并保存到本地
    Args:
        documents (List[Document]): 文档列表
        save_path (str): 向量库保存路径
    Returns:
        FAISS: 构建的 FAISS 向量库
    """
    embedding = get_embedding()

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embedding
    )

    vectorstore.save_local(save_path)
    return vectorstore

def get_embedding() -> HuggingFaceEmbeddings:
    """
    中文法规 / 规范 RAG 专用 embedding
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )

def regulation_to_documents(regulations: List[Regulation]) -> List[Document]:
    """
    将法规条例列表转换为文档列表。
    
    Args:
        regulations (List[Regulation]): 法规条例列表。
        
    Returns:
        List[Document]: 文档列表。
    """
    documents = []
    for reg in regulations:
        doc = Document(
            # 用于 embedding 的文本
            page_content=f"{reg.regulation_id} {reg.content}",
            # 不参与 embedding，但用于溯源和引用
            metadata={
                "regulation_id": reg.regulation_id,
                "article": reg.article,
            }
        )
        documents.append(doc)
    return documents

# 测试代码
if __name__ == "__main__":
    from loader import load_regulation_data
    
    regulations = load_regulation_data(
        "data/《建设工程施工现场供用电安全规范》GB50194-2014.txt"
    )
    documents = regulation_to_documents(regulations)
    vectorstore = build_faiss_vectorstore(documents, save_path="./faiss_index")
    print(f"已构建向量库，条文数量：{len(documents)}")
