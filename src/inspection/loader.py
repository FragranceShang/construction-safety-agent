from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import RulePackItem


DEFAULT_RULEPACK_CANDIDATES = (
    "rulepack.json",
    "gb50194_peidianxiang_rulepack_triggers_dump.json",
)


def resolve_rulepack_path(
    project_root: str | Path = ".",
    explicit_path: str | Path | None = None,
) -> Path:
    """
    优先读取根目录下的 rulepack.json；
    若不存在，则兼容现有 dump 文件命名。
    """
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"未找到指定规则包文件：{explicit_path}")

    root = Path(project_root)
    for name in DEFAULT_RULEPACK_CANDIDATES:
        candidate = root / name
        if candidate.exists():
            return candidate.resolve()

    matches = sorted(root.glob("*rulepack*.json"))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"在 {root.resolve()} 下未找到 rulepack.json 或 *rulepack*.json"
    )


def load_rulepack(
    project_root: str | Path = ".",
    explicit_path: str | Path | None = None,
) -> List[RulePackItem]:
    path = resolve_rulepack_path(project_root=project_root, explicit_path=explicit_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("rulepack 文件格式错误：根节点应为 list")

    return [RulePackItem.model_validate(item) for item in raw]


def rulepack_to_dicts(rulepack: Iterable[RulePackItem]) -> list[dict]:
    return [item.model_dump() for item in rulepack]
