from __future__ import annotations

import json
import os
from typing import Iterable

from inspection.image_focus import build_focus_hint, build_focus_image_paths
from inspection.models import ClauseJudgment, RulePackItem, SceneParseResult
from inspection.prompt import RULE_JUDGE_VLM_PROMPT
from model.inspection_state import InspectionState
from utils.inspection_logger import inspection_logger
from utils.json_utils import safe_load_json
from utils.llm import VISION_JUDGE_MODEL, call_multimodal_llm, get_llm
from utils.wandb import log_metrics


_DEFECT_TERMS = (
    "破损",
    "缺损",
    "缺盖",
    "脱落",
    "碎裂",
    "裂纹",
    "损坏",
    "空缺插座",
    "插座有破损痕迹",
)
_PROTECTION_TERMS = (
    "未见保护措施",
    "无保护措施",
    "裸露导线",
    "疑似裸露导线",
    "未封闭开口",
    "开口可见",
    "线缆下垂",
    "受力",
    "尖锐断口",
)


def _scene_text(scene: SceneParseResult) -> str:
    parts: list[str] = [scene.to_search_text()]
    parts.extend(scene.potential_hazards)
    parts.extend(scene.conditions)
    parts.extend(scene.uncertain_points)
    return "\n".join([part for part in parts if part])


def _hazard_matches(scene_text: str, keywords: Iterable[str]) -> list[str]:
    return [word for word in keywords if word and word in scene_text]


def _heuristic_judge(
    scene: SceneParseResult, rule: RulePackItem, retrieval_score: float
) -> ClauseJudgment:
    """
    无 VLM / dry-run 时的保底规则。
    原则：优先对“图中已经描述得很明确的可见风险”给出明确违规；其余保持保守。
    """
    scope = scene.observation_scope
    visibility = rule.primary_visibility
    scene_text = _scene_text(scene)

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

    if rule.spec_clause == "6.4.1":
        hits = _hazard_matches(scene_text, _DEFECT_TERMS)
        payload["applicability"] = "matched"
        if hits:
            payload["verdict"] = "non_compliant"
            payload["evidence_against"] = [
                "图片描述中出现电器/插座破损、缺损或缺盖线索",
                "条款要求配电箱内电器应完好，不应使用破损电器",
            ]
            payload["reason"] = (
                "当前可见部位存在破损电器线索，已触达条款6.4.1的直接违规点。"
            )
        else:
            payload["verdict"] = "doubtful"
            payload["missing_evidence"] = [
                "虽可见箱内电器，但未见足够近景以全面核验所有电器完好性"
            ]
            payload["reason"] = "未见足够明确的破损证据，也无法全面确认所有电器均完好。"
        return ClauseJudgment.model_validate(payload)

    if rule.spec_clause == "6.3.16":
        hits = _hazard_matches(scene_text, _PROTECTION_TERMS)
        payload["applicability"] = "matched"
        if hits:
            payload["verdict"] = "non_compliant"
            payload["evidence_against"] = [
                "图片描述中出现未封闭开口/裸露导线/线缆缺少保护等线索",
                "条款要求进出线不应承受外力，接触尖锐断口时应有保护措施",
            ]
            payload["reason"] = (
                "进出线区域已出现直接可见的保护不足线索，可支持违规判断。"
            )
        else:
            payload["verdict"] = "doubtful"
            payload["missing_evidence"] = ["缺少更近距离的进出线与固定方式特写"]
            payload["reason"] = "条款与当前画面相关，但证据尚不足以下明确结论。"
        return ClauseJudgment.model_validate(payload)

    if rule.spec_clause == "6.3.13":
        payload["applicability"] = "matched"
        if any(flag in scene_text for flag in ("未封闭开口", "开口可见")):
            payload["verdict"] = "non_compliant"
            payload["evidence_against"] = [
                "进出线/连接器区域存在开口异常或接口缺失线索",
                "当前连接器/开口状态无法体现应有的完整配套与合规安装状态",
            ]
            payload["reason"] = "接口区域存在明显异常开口，已超出正常合规连接器外观。"
        else:
            payload["verdict"] = "doubtful"
            payload["evidence_for"] = ["图片可见配电箱底部工业插座与出线区域"]
            payload["missing_evidence"] = [
                "缺少完整的进线口/出线口位置与连接器合规特写"
            ]
            payload["reason"] = (
                "能看到接口区域，但仍不足以仅凭当前画面确认条款全部要求。"
            )
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


def _vlm_enabled(state: InspectionState) -> bool:
    if state.get("dry_run"):
        return False
    return bool(os.getenv("OPENROUTER_API_KEY"))


def react_judge_node(state: InspectionState) -> InspectionState:
    inspection_logger.info(
        "react_judge_node | question=%s, image_path=%s, candidate_rules=%d",
        state.get("question"),
        state.get("image_path"),
        len(state.get("candidate_rules", [])),
    )

    scene = SceneParseResult.model_validate(state["scene_parse"])
    candidates = state.get("candidate_rules", [])
    question = state.get(
        "question",
        "请按 rulepack 判断这张施工现场图片中可见的施工安全问题。",
    )

    judgments: list[dict] = []
    client = None
    if _vlm_enabled(state):
        client = get_llm()

    for item in candidates:
        rule = RulePackItem.model_validate(item)
        retrieval_score = float(item.get("retrieval_score", 0.0))

        if client is None:
            judgment = _heuristic_judge(scene, rule, retrieval_score)
            judgments.append(judgment.model_dump())
            continue

        focus_hint = build_focus_hint(rule, scene)
        focus_images = build_focus_image_paths(
            image_path=state["image_path"],
            rule=rule,
            scene=scene,
        )
        prompt = RULE_JUDGE_VLM_PROMPT.format(
            focus_hint=focus_hint,
            scene_json=json.dumps(scene.model_dump(), ensure_ascii=False, indent=2),
            question=question,
            rule_json=json.dumps(item, ensure_ascii=False, indent=2),
        )

        try:
            raw = call_multimodal_llm(
                client,
                image_paths=focus_images,
                instruction=prompt,
                model=VISION_JUDGE_MODEL,
                temperature=0.0,
                max_tokens=1400,
            )
            data = safe_load_json(raw, default={})
            if not data:
                raise ValueError("judge vlm 未返回 JSON")
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

    # log_metrics(
    #     {
    #         "initial_judgments": len(judgments),
    #     }
    # )
    return state
