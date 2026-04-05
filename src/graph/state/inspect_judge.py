from __future__ import annotations

import json
import os

from inspection.models import ClauseJudgment, RulePackItem, SceneParseResult
from inspection.prompt import RULE_JUDGE_PROMPT
from model.inspection_state import InspectionState
from utils.json_utils import safe_load_json
from utils.llm import TEXR_MODEL, call_llm, call_vision_llm, get_llm
from utils.wandb import log_metrics


def _heuristic_judge(
    scene: SceneParseResult, rule: RulePackItem, retrieval_score: float
) -> ClauseJudgment:
    """
    无 LLM / dry-run 时的保底规则。
    原则：宁可保守存疑，也不要过度判违。
    """
    scope = scene.observation_scope
    visibility = rule.primary_visibility

    payload = {
        "rulepack_id": rule.rulepack_id,
        "spec_clause": rule.spec_clause,
        "spec_name": rule.spec_name,
        "clause_text": rule.clause_text,
        "visibility_tag": visibility,
        "trigger_name": rule.primary_trigger_name,
        "applicability": "uncertain",
        "verdict": "doubtful",
        "evidence_for": [],
        "evidence_against": [],
        "missing_evidence": [],
        "reason": "离线保底判定：当前仅基于检索与观察边界，默认保守输出存疑。",
        "disposal_suggestion": rule.disposal_suggestion,
        "retrieval_score": retrieval_score,
        "reflection_note": "",
    }

    scene_text = scene.to_search_text()

    if visibility == "台账" and not scope.ledger_available:
        payload["applicability"] = "uncertain"
        payload["verdict"] = "doubtful"
        payload["missing_evidence"] = ["缺少检测记录、台账或系统图等资料证据"]
        payload["reason"] = "该条款依赖台账/记录，单张现场图片无法直接核验。"
        return ClauseJudgment.model_validate(payload)

    if visibility == "内部" and not scope.inside_visible:
        payload["applicability"] = "uncertain"
        payload["verdict"] = "doubtful"
        payload["missing_evidence"] = ["缺少开箱近景或箱内元件细节"]
        payload["reason"] = "该条款依赖箱内电器与接线细节，当前图片不可见。"
        return ClauseJudgment.model_validate(payload)

    if "防雨" in rule.clause_text and ("户外" in scene_text or "露天" in scene_text):
        payload["applicability"] = "matched"
        if any(flag in scene_text for flag in ("未见明显防雨", "缺少防雨", "未见防雨")):
            payload["verdict"] = "non_compliant"
            payload["evidence_against"] = ["图片显示为户外安装，但未见明确防雨措施"]
            payload["reason"] = "户外使用与防雨要求直接相关，且画面存在缺少防雨线索。"
        else:
            payload["verdict"] = "doubtful"
            payload["evidence_for"] = ["图片显示户外配电装置"]
            payload["missing_evidence"] = ["无法确认防护等级或防雨结构细节"]
            payload["reason"] = (
                "能确认户外场景，但无法仅凭当前角度确认是否满足防雨性能。"
            )
        return ClauseJudgment.model_validate(payload)

    if rule.spec_clause == "6.3.16":
        payload["applicability"] = "matched"
        if any(
            flag in scene_text
            for flag in ("裸露导线", "线缆下垂", "未封闭开口", "疑似裸露导线")
        ):
            payload["verdict"] = "non_compliant"
            payload["evidence_against"] = [
                "可见线缆/接口存在暴露、下垂或未封闭开口线索"
            ]
            payload["reason"] = "图中进出线与接口区域存在明显受力/保护不足风险。"
        else:
            payload["verdict"] = "doubtful"
            payload["missing_evidence"] = ["缺少更近距离的进出线与固定方式特写"]
            payload["reason"] = "与当前画面相关，但证据尚不足以下明确结论。"
        return ClauseJudgment.model_validate(payload)

    if rule.spec_clause == "6.3.13":
        payload["applicability"] = "matched"
        payload["verdict"] = "doubtful"
        payload["evidence_for"] = ["图片可见配电箱底部工业插座与出线区域"]
        payload["missing_evidence"] = ["缺少完整的进线口/出线口位置与连接器合规特写"]
        payload["reason"] = (
            "能看到接口区域，但仍不足以仅凭当前画面确认进出线口设置是否完全符合条款。"
        )
        return ClauseJudgment.model_validate(payload)

    if rule.spec_clause == "6.3.15":
        payload["applicability"] = "uncertain"
        if "移动式配电箱" in scene_text and any(
            flag in scene_text for flag in ("非橡套电缆", "硬质电缆", "普通塑料电线")
        ):
            payload["verdict"] = "non_compliant"
            payload["evidence_against"] = ["移动式配电箱使用的线缆疑似并非橡套软电缆"]
            payload["reason"] = "条款直接要求移动式配电箱使用橡套软电缆。"
        else:
            payload["verdict"] = "doubtful"
            payload["missing_evidence"] = ["缺少可确认箱体是否移动式以及线缆材质的近景"]
            payload["reason"] = (
                "当前图片无法可靠确认是否为移动式配电箱，也无法确认线缆材质。"
            )
        return ClauseJudgment.model_validate(payload)

    if any(key in rule.clause_text for key in ("进线", "出线", "承受外力", "保护措施")):
        payload["applicability"] = "matched"
        payload["verdict"] = "doubtful"
        payload["missing_evidence"] = ["缺少更近距离的进出线与固定方式特写"]
        payload["reason"] = "条款与当前画面相关，但证据尚不足以下明确结论。"
        return ClauseJudgment.model_validate(payload)

    return ClauseJudgment.model_validate(payload)


def _llm_enabled(state: InspectionState) -> bool:
    if state.get("dry_run"):
        return False
    return bool(os.getenv("OPENROUTER_API_KEY"))


def react_judge_node(state: InspectionState) -> InspectionState:
    scene = SceneParseResult.model_validate(state["scene_parse"])
    candidates = state.get("candidate_rules", [])
    question = state.get(
        "question",
        "请按 rulepack 判断这张施工现场图片中可见的施工安全问题。",
    )

    judgments: list[dict] = []
    client = None
    if _llm_enabled(state):
        client = get_llm()

    for item in candidates:
        rule = RulePackItem.model_validate(item)
        retrieval_score = float(item.get("retrieval_score", 0.0))

        if client is None:
            judgment = _heuristic_judge(scene, rule, retrieval_score)
            judgments.append(judgment.model_dump())
            continue

        prompt = RULE_JUDGE_PROMPT.format(
            scene_json=json.dumps(scene.model_dump(), ensure_ascii=False, indent=2),
            question=question,
            rule_json=json.dumps(item, ensure_ascii=False, indent=2),
        )

        try:
            raw = call_vision_llm(client, state["image_path"], prompt)
            data = safe_load_json(raw, default={})
            if not data:
                raise ValueError("judge llm 未返回 JSON")
            data.setdefault("rulepack_id", rule.rulepack_id)
            data.setdefault("spec_clause", rule.spec_clause)
            data.setdefault("spec_name", rule.spec_name)
            data.setdefault("clause_text", rule.clause_text)
            data.setdefault("visibility_tag", rule.primary_visibility)
            data.setdefault("trigger_name", rule.primary_trigger_name)
            data.setdefault("disposal_suggestion", rule.disposal_suggestion)
            data.setdefault("retrieval_score", retrieval_score)
            data.setdefault("reflection_note", "")
            judgment = ClauseJudgment.model_validate(data)
        except Exception:
            judgment = _heuristic_judge(scene, rule, retrieval_score)

        judgments.append(judgment.model_dump())

    state["initial_judgments"] = judgments

    log_metrics(
        {
            "initial_judgments": len(judgments),
        }
    )
    return state
