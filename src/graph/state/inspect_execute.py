from __future__ import annotations

import json
import os

from inspection.image_focus import build_action_image_paths
from inspection.models import (
    ActionObservation,
    ClauseJudgment,
    FollowupActionPlan,
    RulePackItem,
    SceneParseResult,
)
from inspection.prompt import ACTION_EXECUTION_PROMPT, FOLLOWUP_ACTION_CATALOG
from model.inspection_state import InspectionState
from utils.inspection_logger import inspection_logger
from utils.json_utils import safe_load_json
from utils.llm import VISION_JUDGE_MODEL, call_multimodal_llm, get_llm
from utils.wandb import log_metrics


_DIRECT_DEFECTS = ("破损", "缺损", "缺盖", "脱落", "碎裂", "裂纹", "损坏")
_DIRECT_PROTECTION = (
    "未封闭开口",
    "开口可见",
    "裸露导线",
    "未见保护措施",
    "线缆下垂",
    "受力",
    "尖锐断口",
)
_RAIN_PROBLEMS = ("未见明显防雨", "缺少防雨", "未见防雨")


def _vlm_enabled(state: InspectionState) -> bool:
    if state.get("dry_run"):
        return False
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _scene_text(scene: SceneParseResult) -> str:
    return "\n".join(
        [
            scene.summary,
            " ".join(scene.visible_texts),
            " ".join(scene.conditions),
            " ".join(scene.potential_hazards),
            " ".join(scene.uncertain_points),
        ]
    )


def _build_instruction(action_type: str) -> str:
    meta = FOLLOWUP_ACTION_CATALOG.get(
        action_type, FOLLOWUP_ACTION_CATALOG["VISUAL_DETAIL"]
    )
    return f"{meta['desc_template']} {meta['expect']}"


def _heuristic_execute(
    scene: SceneParseResult,
    rule: RulePackItem,
    plan: FollowupActionPlan,
    image_paths: list[str],
    roi_regions,
) -> ActionObservation:
    text = _scene_text(scene)
    observations: list[str] = []
    extracted_texts: list[str] = []
    usable_evidence: list[str] = []
    unresolved: list[str] = []

    if plan.action_type == "OCR":
        extracted_texts = list(scene.visible_texts)
        if extracted_texts:
            observations.append("场景解析中存在可读文字，可作为 OCR 的弱替代输出。")
            usable_evidence.extend(extracted_texts[:3])
        else:
            unresolved.append("当前场景未提供足够可读文字，OCR 未带来新增证据。")

    elif plan.action_type == "VISUAL_DETAIL":
        for term in _DIRECT_DEFECTS + _DIRECT_PROTECTION:
            if term in text:
                usable_evidence.append(
                    f"场景解析中出现“{term}”线索，相关局部值得重点复核。"
                )
        if usable_evidence:
            observations.append("局部细节动作命中了场景中的直接风险线索。")
        else:
            unresolved.append("当前场景描述未提供足以改变 verdict 的局部细节线索。")

    elif plan.action_type == "VISUAL_CHECK":
        if any(term in text for term in _RAIN_PROBLEMS):
            usable_evidence.append("户外/露天场景下未见明确防雨线索。")
        if "箱门未关闭" in text:
            usable_evidence.append("场景条件中出现箱门未关闭线索。")
        if scene.environment:
            observations.append(f"环境信息：{'、'.join(scene.environment)}")
        if not usable_evidence:
            unresolved.append("整体外观检查未发现足以改变 verdict 的新增信息。")

    elif plan.action_type == "GEOMETRY":
        if any(term in text for term in ("高度", "距离", "间距", "离地")):
            usable_evidence.append("场景文字中出现几何/距离线索，可供辅助判断。")
        else:
            unresolved.append("缺少可靠参照物，无法仅凭当前场景做几何判断。")

    status = (
        "completed" if usable_evidence or extracted_texts or observations else "no_gain"
    )
    summary = (
        "；".join(usable_evidence[:2])
        if usable_evidence
        else (
            "；".join(unresolved[:1])
            if unresolved
            else "本次 follow-up 未带来新增证据。"
        )
    )

    return ActionObservation(
        action_id=plan.action_id,
        rulepack_id=plan.rulepack_id,
        spec_clause=plan.spec_clause,
        action_type=plan.action_type,
        observability=plan.observability,
        status=status,
        target=plan.target,
        roi_regions=list(roi_regions),
        image_paths=image_paths,
        observations=observations,
        extracted_texts=extracted_texts,
        usable_evidence=usable_evidence,
        unresolved=unresolved,
        summary=summary,
    )


def execute_followup_node(state: InspectionState) -> InspectionState:
    inspection_logger.info(
        "execute_followup_node | followup_plan=%s", state.get("followup_plan")
    )

    scene = SceneParseResult.model_validate(state["scene_parse"])
    rule_map = {
        item["rulepack_id"]: RulePackItem.model_validate(item)
        for item in state.get("candidate_rules", [])
    }
    judgment_map = {
        item["rulepack_id"]: ClauseJudgment.model_validate(item)
        for item in state.get("initial_judgments", [])
    }
    client = get_llm() if _vlm_enabled(state) else None

    observations: list[dict] = []
    for item in state.get("followup_plan", []):
        plan = FollowupActionPlan.model_validate(item)
        rule = rule_map.get(plan.rulepack_id)
        if not rule:
            continue
        judgment = judgment_map.get(plan.rulepack_id)
        if plan.observability != "same_image_recoverable":
            obs = ActionObservation(
                action_id=plan.action_id,
                rulepack_id=plan.rulepack_id,
                spec_clause=plan.spec_clause,
                action_type=plan.action_type,
                observability=plan.observability,
                status="skipped",
                target=plan.target,
                roi_regions=[],
                image_paths=[state["image_path"]],
                observations=[],
                extracted_texts=[],
                usable_evidence=[],
                unresolved=[
                    f"该条款当前被规划为 {plan.observability}，不执行同图补证。"
                ],
                summary=f"跳过：{plan.observability}",
            )
            observations.append(obs.model_dump())
            continue

        image_paths, roi_regions = build_action_image_paths(
            image_path=state["image_path"],
            rule=rule,
            scene=scene,
            action_plan=plan,
            client=client,
            judgment=judgment,
            output_dir="outputs/followup_focus",
        )

        if client is None:
            obs = _heuristic_execute(scene, rule, plan, image_paths, roi_regions)
            observations.append(obs.model_dump())
            continue

        prompt = ACTION_EXECUTION_PROMPT.format(
            scene_json=json.dumps(scene.model_dump(), ensure_ascii=False, indent=2),
            rule_json=json.dumps(rule.model_dump(), ensure_ascii=False, indent=2),
            action_json=json.dumps(plan.model_dump(), ensure_ascii=False, indent=2),
            action_instruction=_build_instruction(plan.action_type),
        )
        try:
            raw = call_multimodal_llm(
                client,
                image_paths=image_paths,
                instruction=prompt,
                model=VISION_JUDGE_MODEL,
                temperature=0.0,
                max_tokens=1200,
            )
            data = safe_load_json(raw, default={}) or {}
            data.setdefault("action_id", plan.action_id)
            data.setdefault("rulepack_id", plan.rulepack_id)
            data.setdefault("spec_clause", plan.spec_clause)
            data.setdefault("action_type", plan.action_type)
            data.setdefault("observability", plan.observability)
            data.setdefault("target", plan.target)
            data.setdefault(
                "roi_regions", [region.model_dump() for region in roi_regions]
            )
            data.setdefault("image_paths", image_paths)
            obs = ActionObservation.model_validate(data)
        except Exception:
            obs = _heuristic_execute(scene, rule, plan, image_paths, roi_regions)
        observations.append(obs.model_dump())

    state["action_observations"] = observations
    # log_metrics(
    #     {
    #         "action_observations": len(observations),
    #         "executed_actions": sum(
    #             1 for item in observations if item.get("status") == "completed"
    #         ),
    #     }
    # )
    return state
