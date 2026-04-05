import base64
import os
from dotenv import load_dotenv
from openai import OpenAI

TEXR_MODEL = "nex-agi/deepseek-v3.1-nex-n1"
VISION_MODEL = "qwen/qwen3-vl-8b-instruct"


def get_llm():
    """
    延迟导入 OpenAI 客户端，便于在 dry-run / 单元测试场景下不安装 openai 也能导入项目。
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "当前环境未安装 openai 依赖。若需调用真实模型，请先安装 requirements.txt。"
        ) from exc

    load_dotenv()
    api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("未在环境变量中找到 OPENROUTER_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def call_llm(
    client,
    prompt: str,
    model: str = TEXR_MODEL,
    temperature: float = 0.2,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        return "".join(str(item) for item in content).strip()
    return str(content).strip()


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision_llm(client: OpenAI, image_path: str, instruction: str) -> str:
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0.0,
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

    content = response.choices[0].message.content
    if isinstance(content, list):
        return "".join(str(item) for item in content).strip()
    return str(content)
