import base64
import mimetypes
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

TEXR_MODEL = "nex-agi/deepseek-v3.1-nex-n1"
VISION_MODEL = "qwen/qwen3-vl-8b-instruct"
VISION_JUDGE_MODEL = os.getenv("VISION_JUDGE_MODEL", VISION_MODEL)


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


def _normalize_content(content) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def call_llm(
    client,
    prompt: str,
    model: str = TEXR_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return _normalize_content(response.choices[0].message.content)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_content_item(image_path: str) -> dict:
    base64_image = encode_image(image_path)
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        suffix = Path(image_path).suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
    }


def call_multimodal_llm(
    client,
    image_paths: Iterable[str],
    instruction: str,
    model: str = VISION_JUDGE_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1600,
) -> str:
    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in image_paths:
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    content = [{"type": "text", "text": instruction}]
    content.extend(_image_content_item(path) for path in unique_paths)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_tokens=max_tokens,
    )
    return _normalize_content(response.choices[0].message.content)


def call_vision_llm(client, image_path: str, instruction: str) -> str:
    return call_multimodal_llm(
        client,
        image_paths=[image_path],
        instruction=instruction,
        model=VISION_MODEL,
        temperature=0.0,
        max_tokens=1500,
    )
