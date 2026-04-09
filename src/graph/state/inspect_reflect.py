from __future__ import annotations

import json
import os

from inspection.image_focus import build_focus_hint, build_focus_image_paths
from inspection.models import ClauseJudgment, RulePackItem, SceneParseResult
from inspection.prompt import REFLECTION_VLM_PROMPT
from model.inspection_state import InspectionState
from utils.inspection_logger import inspection_logger
from utils.json_utils import safe_load_json
from utils.llm import VISION_JUDGE_MODEL, call_multimodal_llm, get_llm
from utils.wandb import log_metrics


def _apply_guardrails(
    scene: SceneParseResult,
    rule: RulePackItem,
    judgment: ClauseJudgment,
) -> ClauseJudgment:
    scope = scene.observation_scope
    updated = judgment.model_copy(deep=True)

    if updated.applicability == "unmatched":
        updated.verdict = "not_applicable"
        updated.reflection_note = "复核后认为当前图片与该条款场景不匹配，调整为不适用。"
        return updated

    if rule.primary_visibility == "内部" and not scope.inside_visible:
        if updated.verdict in {"compliant", "non_compliant"}:
            updated.verdict = "doubtful"
            updated.reflection_note = (
                "该条款需内部可见，但当前图片未提供足够箱内细节，已降级为存疑。"
            )
        if "缺少开箱近景或箱内元件细节" not in updated.missing_evidence:
            updated.missing_evidence.append("缺少开箱近景或箱内元件细节")
        return updated

    if rule.primary_visibility == "台账" and not scope.ledger_available:
        if updated.verdict in {"compliant", "non_compliant"}:
            updated.verdict = "doubtful"
            updated.reflection_note = (
                "该条款依赖台账/记录证据，单张图片不足以形成强结论，已降级为存疑。"
            )
        if "缺少检测记录、台账或系统图等资料证据" not in updated.missing_evidence:
            updated.missing_evidence.append("缺少检测记录、台账或系统图等资料证据")
        return updated

    if (
        (not scope.parameter_readable)
        and updated.verdict in {"compliant", "non_compliant"}
        and any(
            key in rule.clause_text.lower() for key in ("ip", "30ma", "0.1s", "额定值")
        )
    ):
        updated.verdict = "doubtful"
        updated.reflection_note = (
            "该条款涉及参数读取，但当前图片参数不可读，已降级为存疑。"
        )
        if "缺少可读参数近景" not in updated.missing_evidence:
            updated.missing_evidence.append("缺少可读参数近景")
        return updated

    if updated.verdict == "non_compliant" and not updated.evidence_against:
        updated.verdict = "doubtful"
        updated.reflection_note = "未提供直接、可复核的反向证据，违规结论已降级为存疑。"
        if "缺少直接反向证据" not in updated.missing_evidence:
            updated.missing_evidence.append("缺少直接反向证据")
        return updated

    if updated.verdict == "compliant" and not updated.evidence_for:
        updated.verdict = "doubtful"
        updated.reflection_note = "未提供足够支持证据，符合结论已降级为存疑。"
        if "缺少支持结论的可复核证据" not in updated.missing_evidence:
            updated.missing_evidence.append("缺少支持结论的可复核证据")
        return updated

    return updated


def _vlm_enabled(state: InspectionState) -> bool:
    if state.get("dry_run"):
        return False
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _rule_from_candidate_or_judgment(
    candidate: dict | None, judgment: ClauseJudgment
) -> RulePackItem:
    if candidate:
        return RulePackItem.model_validate(candidate)
    return RulePackItem.model_validate(
        {
            "rulepack_id": judgment.rulepack_id,
            "spec_code": "",
            "spec_clause": judgment.spec_clause,
            "spec_name": judgment.spec_name,
            "clause_text": judgment.clause_text,
            "judge_dimension": "",
            "judge_boundary": {
                "compliant": "",
                "non_compliant": "",
                "doubtful": "",
            },
            "keywords": [],
            "similar_words": [],
            "triggers": [
                {
                    "trigger_id": "",
                    "trigger_name": judgment.trigger_name,
                    "visibility_tag": judgment.visibility_tag,
                }
            ],
            "disposal_suggestion": judgment.disposal_suggestion,
        }
    )


def reflect_judgments_node(state: InspectionState) -> InspectionState:
    inspection_logger.info(
        "reflect_judgments_node | judgments=%d, candidate_rules=%d",
        len(state.get("initial_judgments", [])),
        len(state.get("candidate_rules", [])),
    )

    scene = SceneParseResult.model_validate(state["scene_parse"])
    client = get_llm() if _vlm_enabled(state) else None

    source_items = state.get("rejudge_judgments") or state.get("initial_judgments", [])
    final_judgments: list[dict] = []
    for item in source_items:
        judgment = ClauseJudgment.model_validate(item)
        candidate = next(
            (
                cand
                for cand in state.get("candidate_rules", [])
                if cand.get("rulepack_id") == judgment.rulepack_id
            ),
            None,
        )
        rule = _rule_from_candidate_or_judgment(candidate, judgment)

        guarded = _apply_guardrails(scene, rule, judgment)
        if client is None or guarded.verdict not in {"compliant", "non_compliant"}:
            final_judgments.append(guarded.model_dump())
            continue

        focus_hint = build_focus_hint(rule, scene)
        focus_images = build_focus_image_paths(
            image_path=state["image_path"],
            rule=rule,
            scene=scene,
        )
        prompt = REFLECTION_VLM_PROMPT.format(
            focus_hint=focus_hint,
            scene_json=json.dumps(scene.model_dump(), ensure_ascii=False, indent=2),
            rule_json=json.dumps(
                candidate or rule.model_dump(), ensure_ascii=False, indent=2
            ),
            judgment_json=json.dumps(
                guarded.model_dump(), ensure_ascii=False, indent=2
            ),
        )

        try:
            raw = call_multimodal_llm(
                client,
                image_paths=focus_images,
                instruction=prompt,
                model=VISION_JUDGE_MODEL,
                temperature=0.0,
                max_tokens=1200,
            )
            data = safe_load_json(raw, default={}) or {}
            revised = guarded.model_dump()
            revised.update(data)
            revised.setdefault("spec_name", guarded.spec_name)
            revised.setdefault("clause_text", guarded.clause_text)
            revised.setdefault("visibility_tag", guarded.visibility_tag)
            revised.setdefault("trigger_name", guarded.trigger_name)
            revised.setdefault("disposal_suggestion", guarded.disposal_suggestion)
            revised.setdefault("retrieval_score", guarded.retrieval_score)
            final = ClauseJudgment.model_validate(revised)
            final = _apply_guardrails(scene, rule, final)
        except Exception:
            final = guarded

        final_judgments.append(final.model_dump())

    state["final_judgments"] = final_judgments
    # log_metrics({"final_judgments": len(final_judgments)})
    return state
