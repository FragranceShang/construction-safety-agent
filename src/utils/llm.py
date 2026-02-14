import base64
from openai import OpenAI
from dotenv import load_dotenv
import os

TEXR_MODEL = "nex-agi/deepseek-v3.1-nex-n1"

VISION_MODEL = "qwen/qwen3-vl-8b-instruct"


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
    model: str = TEXR_MODEL,
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
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


def encode_image(image_path: str) -> str:
    """把本地图片转为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision_llm(client: OpenAI, image_path: str, instruction: str) -> str:
    """
    使用 Qwen-VL 解析图片，返回文本
    """

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0.0,  # 视觉抽取必须 0，避免幻觉
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=1500,
    )

    return response.choices[0].message.content
