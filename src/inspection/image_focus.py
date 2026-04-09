from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from PIL import Image

from inspection.prompt import ROI_PROPOSAL_PROMPT
from utils.json_utils import safe_load_json
from utils.llm import VISION_JUDGE_MODEL, call_multimodal_llm

from .models import ClauseJudgment, FollowupActionPlan, RoiRegion, RulePackItem, SceneParseResult


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


def _crop_specs(rule: RulePackItem, scene: SceneParseResult, action_type: str | None = None) -> list[tuple[str, tuple[float, float, float, float]]]:
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
                action_type or "",
            ]
        )
    )

    specs: list[tuple[str, tuple[float, float, float, float]]] = []
    if action_type == "OCR":
        specs.append(("center", (0.10, 0.08, 0.90, 0.90)))
        specs.append(("top_center", (0.06, 0.00, 0.94, 0.65)))
    elif action_type == "GEOMETRY":
        specs.append(("full", (0.0, 0.0, 1.0, 1.0)))
    elif any(key.lower() in text for key in _SOCKET_KEYS):
        specs.append(("bottom", (0.0, 0.38, 1.0, 1.0)))
        specs.append(("bottom_center", (0.18, 0.34, 0.82, 0.96)))
    elif any(key.lower() in text for key in _INTERNAL_KEYS):
        specs.append(("center", (0.10, 0.08, 0.90, 0.86)))
        specs.append(("top_center", (0.08, 0.00, 0.92, 0.72)))
    elif any(key.lower() in text for key in _LABEL_KEYS):
        specs.append(("center", (0.10, 0.08, 0.90, 0.90)))
    else:
        specs.append(("center", (0.10, 0.08, 0.90, 0.90)))

    return specs[:2]


def _sanitize_region(region: RoiRegion) -> RoiRegion:
    x1 = max(0.0, min(0.98, region.x1))
    y1 = max(0.0, min(0.98, region.y1))
    x2 = max(x1 + 0.05, min(1.0, region.x2))
    y2 = max(y1 + 0.05, min(1.0, region.y2))
    return RoiRegion(
        name=region.name or "roi",
        x1=round(x1, 4),
        y1=round(y1, 4),
        x2=round(x2, 4),
        y2=round(y2, 4),
        reason=region.reason,
    )


def heuristic_roi_regions(
    image_path: str,
    rule: RulePackItem,
    scene: SceneParseResult,
    action_type: str | None = None,
) -> list[RoiRegion]:
    del image_path
    specs = _crop_specs(rule, scene, action_type=action_type)
    return [
        RoiRegion(name=name, x1=box[0], y1=box[1], x2=box[2], y2=box[3], reason="heuristic_fallback")
        for name, box in specs
    ]


def propose_roi_regions(
    *,
    client,
    image_path: str,
    rule: RulePackItem,
    scene: SceneParseResult,
    judgment: ClauseJudgment | None = None,
    action_plan: FollowupActionPlan | None = None,
) -> list[RoiRegion]:
    if client is None:
        return heuristic_roi_regions(image_path, rule, scene, action_type=action_plan.action_type if action_plan else None)

    prompt = ROI_PROPOSAL_PROMPT.format(
        scene_json=json.dumps(scene.model_dump(), ensure_ascii=False, indent=2),
        rule_json=json.dumps(rule.model_dump(), ensure_ascii=False, indent=2),
        judgment_json=json.dumps(judgment.model_dump(), ensure_ascii=False, indent=2) if judgment else "{}",
        action_json=json.dumps(action_plan.model_dump(), ensure_ascii=False, indent=2) if action_plan else "{}",
    )
    try:
        raw = call_multimodal_llm(
            client,
            image_paths=[image_path],
            instruction=prompt,
            model=VISION_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=900,
        )
        data = safe_load_json(raw, default={}) or {}
        regions_raw = data.get("regions") or []
        regions: list[RoiRegion] = []
        for item in regions_raw:
            try:
                region = _sanitize_region(RoiRegion.model_validate(item))
                regions.append(region)
            except Exception:
                continue
        if regions:
            return regions[:2]
    except Exception:
        pass
    return heuristic_roi_regions(image_path, rule, scene, action_type=action_plan.action_type if action_plan else None)


def crop_image_regions(
    image_path: str,
    regions: Iterable[RoiRegion],
    output_dir: str,
    prefix: str,
) -> list[str]:
    origin = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = [str(origin)]
    with Image.open(origin) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        for idx, region in enumerate(regions, start=1):
            x1 = max(0, min(width - 1, int(width * region.x1)))
            y1 = max(0, min(height - 1, int(height * region.y1)))
            x2 = max(x1 + 1, min(width, int(width * region.x2)))
            y2 = max(y1 + 1, min(height, int(height * region.y2)))
            crop = rgb.crop((x1, y1, x2, y2))
            output_path = out_dir / f"{prefix}_roi_{idx}.jpg"
            crop.save(output_path, format="JPEG", quality=92)
            image_paths.append(str(output_path))

    return image_paths


def build_action_image_paths(
    *,
    image_path: str,
    rule: RulePackItem,
    scene: SceneParseResult,
    action_plan: FollowupActionPlan,
    client=None,
    judgment: ClauseJudgment | None = None,
    output_dir: str = "outputs/followup_focus",
) -> tuple[list[str], list[RoiRegion]]:
    regions = propose_roi_regions(
        client=client,
        image_path=image_path,
        rule=rule,
        scene=scene,
        judgment=judgment,
        action_plan=action_plan,
    )
    prefix = f"{Path(image_path).stem}_{rule.spec_clause.replace('.', '_')}_{action_plan.action_id}"
    image_paths = crop_image_regions(
        image_path=image_path,
        regions=regions,
        output_dir=output_dir,
        prefix=prefix,
    )
    return image_paths, regions


def build_focus_image_paths(
    image_path: str,
    rule: RulePackItem,
    scene: SceneParseResult,
    output_dir: str = "outputs/focus_images",
) -> list[str]:
    """
    兼容首轮 judge / reflect 的旧接口。
    返回 [原图, 裁剪图1, 裁剪图2...]。
    """
    regions = heuristic_roi_regions(image_path, rule, scene)
    prefix = f"{Path(image_path).stem}_{rule.spec_clause.replace('.', '_')}"
    return crop_image_regions(
        image_path=image_path,
        regions=regions,
        output_dir=output_dir,
        prefix=prefix,
    )
