from openai import OpenAI
from dotenv import load_dotenv
import os

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
