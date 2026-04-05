import json
import re
from typing import Any


def safe_load_json(text: str | None, default: Any = None) -> Any:
    """
    尽量从 LLM 输出中提取 JSON。
    支持：
    - 纯 JSON
    - ```json fenced block
    - 带解释文字的 JSON 对象 / 数组
    """
    if text is None:
        return default

    cleaned = re.sub(r"```json|```", "", text).strip()
    if not cleaned:
        return default

    # 1) 直接解析
    for candidate in (cleaned,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 2) 提取对象 / 数组
    matches: list[str] = []
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if obj_match:
        matches.append(obj_match.group())
    if arr_match:
        matches.append(arr_match.group())

    for candidate in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return default


def dump_json(data: Any, path: str, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
