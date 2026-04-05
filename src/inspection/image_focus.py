from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .models import RulePackItem, SceneParseResult


_SOCKET_KEYS = (
    "插座",
    "工业插座",
    "连接器",
    "耦合器",
    "进线",
    "出线",
    "电缆",
    "线缆",
    "受力",
    "保护措施",
    "断口",
)

_INTERNAL_KEYS = (
    "断路器",
    "隔离开关",
    "漏保",
    "漏电",
    "仪表",
    "接线端子",
    "汇流排",
    "n)",
    "pe",
    "铜排",
    "接线",
    "导线",
)

_LABEL_KEYS = (
    "标识",
    "编号",
    "名称",
    "系统图",
    "警示",
    "铭牌",
    "参数",
    "30ma",
    "0.1s",
    "ip",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def build_focus_hint(rule: RulePackItem, scene: SceneParseResult) -> str:
    text = _normalize(
        " ".join(
            [
                rule.clause_text,
                rule.judge_dimension,
                rule.primary_trigger_name,
                scene.inspection_target,
                scene.summary,
                " ".join(scene.potential_hazards),
            ]
        )
    )

    hints: list[str] = []
    if any(key.lower() in text for key in _SOCKET_KEYS):
        hints.append("重点查看箱体下沿的工业插座、进出线口、开孔、线缆护套与受力状态。")
    if any(key.lower() in text for key in _INTERNAL_KEYS):
        hints.append("重点查看箱内断路器、接线端子、汇流排、导线与器件完好性。")
    if any(key.lower() in text for key in _LABEL_KEYS):
        hints.append("重点查看门体/器件附近是否存在可读标识、编号、参数或警示文字。")
    if not hints:
        hints.append("先看全图，再根据条款相关部位放大核对可见证据。")
    return " ".join(hints)


def _crop_specs(rule: RulePackItem, scene: SceneParseResult) -> list[tuple[str, tuple[float, float, float, float]]]:
    text = _normalize(
        " ".join(
            [
                rule.clause_text,
                rule.judge_dimension,
                rule.primary_trigger_name,
                scene.inspection_target,
                scene.summary,
                " ".join(scene.visible_objects),
                " ".join(scene.potential_hazards),
            ]
        )
    )

    specs: list[tuple[str, tuple[float, float, float, float]]] = []
    if any(key.lower() in text for key in _SOCKET_KEYS):
        specs.append(("bottom", (0.0, 0.38, 1.0, 1.0)))
        specs.append(("bottom_center", (0.18, 0.34, 0.82, 0.96)))
    elif any(key.lower() in text for key in _INTERNAL_KEYS):
        specs.append(("center", (0.10, 0.08, 0.90, 0.86)))
        specs.append(("top_center", (0.08, 0.00, 0.92, 0.72)))
    elif any(key.lower() in text for key in _LABEL_KEYS):
        specs.append(("center", (0.10, 0.08, 0.90, 0.90)))
    else:
        specs.append(("center", (0.10, 0.08, 0.90, 0.90)))

    # 最多保留两个局部视图，控制成本
    return specs[:2]


def build_focus_image_paths(
    image_path: str,
    rule: RulePackItem,
    scene: SceneParseResult,
    output_dir: str = "outputs/focus_images",
) -> list[str]:
    """
    返回 [原图, 裁剪图1, 裁剪图2...]。
    裁剪图由条款与场景自动生成，供 VLM 在逐条核验时聚焦细节。
    """
    origin = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = [str(origin)]
    specs = _crop_specs(rule, scene)

    with Image.open(origin) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        for name, (x1r, y1r, x2r, y2r) in specs:
            x1 = max(0, min(width - 1, int(width * x1r)))
            y1 = max(0, min(height - 1, int(height * y1r)))
            x2 = max(x1 + 1, min(width, int(width * x2r)))
            y2 = max(y1 + 1, min(height, int(height * y2r)))
            crop = rgb.crop((x1, y1, x2, y2))
            output_path = out_dir / f"{origin.stem}_{rule.spec_clause.replace('.', '_')}_{name}.jpg"
            crop.save(output_path, format="JPEG", quality=92)
            result.append(str(output_path))

    return result
