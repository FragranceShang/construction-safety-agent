from typing import List
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils.logger import setup_logger
from prompt import REGULATION_QA_PROMPT
from langchain_core.retrievers import BaseRetriever

logger = setup_logger("rag")

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

def get_llm() -> OpenAI:
    """
    获取 OpenAI LLM 实例
    """
    load_dotenv()
    api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("未在环境变量中找到 OPENROUTER_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

def call_llm(
    client: OpenAI,
    prompt: str,
    model: str = "nex-agi/deepseek-v3.1-nex-n1:free",
    temperature: float = 0.2,
) -> str:
    """
    调用大语言模型生成回答。

    Args:
        client (OpenAI):
            OpenRouter OpenAI 客户端。
        prompt (str):
            已格式化完成的 Prompt 文本。
        model (str):
            使用的模型名称。
        temperature (float):
            生成温度。

    Returns:
        str:
            模型生成的回答。
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()

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
if __name__ == "__main__":
    from graph.retriever import load_vectorstore, build_retriever
    vectorstore = load_vectorstore("faiss_index")
    retriever = build_retriever(vectorstore, top_k=9)
    with open("test_results.txt", "w", encoding="utf-8") as f:

        f.write("\n===测试基本检索能力===\n")
        for question in [
            "配电箱的安装高度有什么要求？",
            "架空线路跨越道路时的最小垂直距离是多少？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")

        f.write("\n===测试理解和解释能力===\n")
        for question in [
            "为什么行灯变压器不能带入金属容器内使用？",
            "多级剩余电流保护器如何设置动作电流和时间？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")

        f.write("\n===测试实际应用能力===\n")
        for question in [
            "在脚手架上作业时，对外电线路的安全距离有什么要求？",
            "雨季施工需要注意哪些用电安全事项？",
            "塔式起重机接地应该如何设置？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")

        f.write("\n===测试深度分析能力===\n")
        for question in [
            "手持式电动工具在一般场所和潮湿场所的要求有什么不同？",
            "临时照明和行灯在电压要求上有何区别？为什么？",
            "易燃易爆环境和腐蚀环境对电缆敷设要求有何异同？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")

        f.write("\n===测试实际工作应用===\n")
        for question in [
            "检查配电箱时需要关注哪些关键点？",
            "直埋电缆施工需要满足哪些技术要求？",
            "外电线路附近施工应采取哪些防护措施？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】")
            f.write(result)
            f.write("\n")

        f.write("\n===测试系统健壮性===\n")
        for question in [
            "本规范适用于井下工程吗",
            "规范中对高原环境的特殊要求是什么？",
            "焊接机械的二次线为什么不能用钢筋代替？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")
