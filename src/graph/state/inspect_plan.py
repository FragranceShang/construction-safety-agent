from __future__ import annotations

import json
import os
from collections import Counter

from inspection.models import (
    ClauseJudgment,
    FollowupActionPlan,
    RulePackItem,
    SceneParseResult,
)
from inspection.prompt import (
    ACTION_CATALOG_JSON,
    FOLLOWUP_ACTION_CATALOG,
    FOLLOWUP_PLAN_PROMPT,
)
from model.inspection_state import InspectionState
from utils.inspection_logger import inspection_logger
from utils.json_utils import safe_load_json
from utils.llm import VISION_JUDGE_MODEL, call_multimodal_llm, get_llm
from utils.wandb import log_metrics


_ACTION_ORDER = ("VISUAL_DETAIL", "VISUAL_CHECK", "OCR", "GEOMETRY")


def _vlm_enabled(state: InspectionState) -> bool:
    if state.get("dry_run"):
        return False
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _combined_text(
    scene: SceneParseResult, rule: RulePackItem, judgment: ClauseJudgment
) -> str:
    return "\n".join(
        [
            scene.to_search_text(),
            rule.clause_text,
            rule.judge_dimension,
            rule.primary_trigger_name,
            " ".join(judgment.missing_evidence),
            judgment.reason,
        ]
    )


def _pick_action_type(
    scene: SceneParseResult, rule: RulePackItem, judgment: ClauseJudgment
) -> str:
    text = _combined_text(scene, rule, judgment)
    score_map = Counter()
    for action_type, meta in FOLLOWUP_ACTION_CATALOG.items():
        for keyword in meta["keywords"]:
            if keyword and keyword in text:
                score_map[action_type] += 1

    clause = rule.clause_text
    if any(
        term in clause
        for term in (
            "破损",
            "裸露",
            "接线",
            "端子",
            "汇流排",
            "进线",
            "出线",
            "保护措施",
        )
    ):
        score_map["VISUAL_DETAIL"] += 4
    if any(term in clause for term in ("防雨", "门", "锁", "外壳", "安装", "环境")):
        score_map["VISUAL_CHECK"] += 3
    if any(
        term in clause
        for term in (
            "编号",
            "名称",
            "警示",
            "参数",
            "铭牌",
            "IP",
            "30mA",
            "0.1s",
            "额定值",
        )
    ):
        score_map["OCR"] += 4
    if any(term in clause for term in ("高度", "距离", "离地", "间距", "空间")):
        score_map["GEOMETRY"] += 4

    if not score_map:
        return "VISUAL_DETAIL"
    return sorted(
        score_map.items(),
        key=lambda item: (
            -item[1],
            _ACTION_ORDER.index(item[0]) if item[0] in _ACTION_ORDER else 99,
        ),
    )[0][0]


def _pick_target(action_type: str, rule: RulePackItem) -> str:
    clause = rule.clause_text
    if action_type == "OCR":
        if any(
            term in clause for term in ("警示", "编号", "名称", "系统图", "分路标记")
        ):
            return "箱门或门体附近的标识、编号、警示与系统图区域"
        return "器件铭牌、参数、标签或警示文字区域"
    if action_type == "VISUAL_CHECK":
        if any(term in clause for term in ("防雨", "防潮")):
            return "箱门、外壳、遮挡及周围环境"
        return "设备整体外观、门体状态和安装环境"
    if action_type == "GEOMETRY":
        return "设备与地面或周边参照物的相对位置"
    if any(term in clause for term in ("断路器", "端子", "汇流排", "接线")):
        return "箱内断路器、接线端子和汇流排区域"
    return "底部插座、出线口、开孔和线缆局部"


def _infer_observability(
    scene: SceneParseResult,
    rule: RulePackItem,
    judgment: ClauseJudgment,
    action_type: str,
) -> str:
    need_text = " ".join(
        judgment.missing_evidence
        + [judgment.reason, rule.clause_text, rule.judge_dimension]
    )
    if any(term in need_text for term in ("台账", "记录", "证书", "系统图")):
        return "needs_document"
    if rule.primary_visibility == "内部" and not scene.observation_scope.inside_visible:
        return "needs_new_view"
    if action_type == "OCR" and not (
        scene.observation_scope.parameter_readable
        or scene.observation_scope.door_label_readable
        or scene.visible_texts
    ):
        return "needs_new_view"
    if any(term in need_text for term in ("橡套软电缆", "线缆材质")):
        return "needs_new_view"
    if any(
        term in need_text
        for term in (
            "额定值",
            "独立保护电器",
            "一一对应",
            "端子数量",
            "汇流排",
            "回路对应",
            "分路标记",
        )
    ):
        if action_type == "OCR" and (
            scene.observation_scope.parameter_readable
            or scene.observation_scope.door_label_readable
        ):
            return "same_image_recoverable"
        return "needs_new_view"
    if action_type in {"VISUAL_DETAIL", "VISUAL_CHECK", "GEOMETRY"}:
        return "same_image_recoverable"
    return "not_worth_retry"


def _heuristic_plan(
    scene: SceneParseResult, rule: RulePackItem, judgment: ClauseJudgment, idx: int
) -> list[FollowupActionPlan]:
    action_type = _pick_action_type(scene, rule, judgment)
    observability = _infer_observability(scene, rule, judgment, action_type)
    meta = FOLLOWUP_ACTION_CATALOG[action_type]
    target = _pick_target(action_type, rule)
    why = judgment.reason or "首轮证据不足，需要定向补证。"
    roi_request = target.replace("区域", "")
    stop_if = {
        "OCR": "若已读到参数、标识或警示文字且足以支持/否定条款，则停止。",
        "VISUAL_DETAIL": "若已明确看到破损、裸露、异常开口或连接异常，则停止。",
        "VISUAL_CHECK": "若已明确看到门体状态、防雨状态或整体破损，则停止。",
        "GEOMETRY": "若已有足够参照物支持距离/高度判断，则停止。",
    }[action_type]
    plan = FollowupActionPlan(
        action_id=f"followup_{idx:03d}",
        rulepack_id=rule.rulepack_id,
        spec_clause=rule.spec_clause,
        action_type=action_type,
        observability=observability,  # type: ignore[arg-type]
        target=target,
        why=why,
        expected=meta["expect"],
        roi_request=roi_request,
        stop_if=stop_if,
        priority=1,
    )
    return [plan]


def _vlm_plan(
    scene: SceneParseResult,
    rule: RulePackItem,
    judgment: ClauseJudgment,
    idx: int,
    client,
    image_path: str,
) -> list[FollowupActionPlan]:
    prompt = FOLLOWUP_PLAN_PROMPT.format(
        scene_json=json.dumps(scene.model_dump(), ensure_ascii=False, indent=2),
        rule_json=json.dumps(rule.model_dump(), ensure_ascii=False, indent=2),
        judgment_json=json.dumps(judgment.model_dump(), ensure_ascii=False, indent=2),
        action_catalog_json=ACTION_CATALOG_JSON,
    )
    try:
        raw = call_multimodal_llm(
            client,
            image_paths=[image_path],
            instruction=prompt,
            model=VISION_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=1200,
        )
        data = safe_load_json(raw, default={}) or {}
        observability = data.get("observability") or "not_worth_retry"
        actions = data.get("actions") or []
        plans: list[FollowupActionPlan] = []
        for action_idx, action in enumerate(actions, start=1):
            action_type = action.get("action_type") or _pick_action_type(
                scene, rule, judgment
            )
            meta = FOLLOWUP_ACTION_CATALOG.get(
                action_type, FOLLOWUP_ACTION_CATALOG["VISUAL_DETAIL"]
            )
            plan = FollowupActionPlan(
                action_id=f"followup_{idx:03d}_{action_idx}",
                rulepack_id=rule.rulepack_id,
                spec_clause=rule.spec_clause,
                action_type=action_type,
                observability=observability,
                target=action.get("target") or _pick_target(action_type, rule),
                why=action.get("why") or data.get("reason") or judgment.reason,
                expected=action.get("expected") or meta["expect"],
                roi_request=action.get("roi_request")
                or _pick_target(action_type, rule),
                stop_if=action.get("stop_if")
                or "若已获得足以改变 verdict 的直接证据，则停止。",
                priority=int(action.get("priority") or action_idx),
            )
            plans.append(plan)
        if plans:
            return plans[:2]
    except Exception:
        pass
    return _heuristic_plan(scene, rule, judgment, idx)


def plan_followup_node(state: InspectionState) -> InspectionState:
    inspection_logger.info(
        "plan_followup_node | question=%s, image_path=%s, candidate_rules=%d",
        state.get("question"),
        state.get("image_path"),
        len(state.get("candidate_rules", [])),
    )

    scene = SceneParseResult.model_validate(state["scene_parse"])
    candidate_map = {
        item["rulepack_id"]: RulePackItem.model_validate(item)
        for item in state.get("candidate_rules", [])
    }
    client = get_llm() if _vlm_enabled(state) else None

    plans: list[dict] = []
    idx = 1
    for item in state.get("initial_judgments", []):
        judgment = ClauseJudgment.model_validate(item)
        if judgment.verdict != "doubtful":
            continue
        rule = candidate_map.get(judgment.rulepack_id)
        if not rule:
            continue
        if client is None:
            generated = _heuristic_plan(scene, rule, judgment, idx)
        else:
            generated = _vlm_plan(
                scene, rule, judgment, idx, client, state["image_path"]
            )
        plans.extend(plan.model_dump() for plan in generated)
        idx += 1

    state["followup_plan"] = plans
    # log_metrics(
    #     {
    #         "followup_plan": len(plans),
    #         "same_image_recoverable": sum(
    #             1
    #             for item in plans
    #             if item.get("observability") == "same_image_recoverable"
    #         ),
    #     }
    # )
    return state
