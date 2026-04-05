from __future__ import annotations

import json
import os

from inspection.models import ClauseJudgment, RulePackItem, SceneParseResult
from inspection.prompt import REFLECTION_PROMPT
from model.inspection_state import InspectionState
from utils.json_utils import safe_load_json
from utils.llm import TEXR_MODEL, call_llm, get_llm
from utils.wandb import log_metrics


def _apply_guardrails(
    scene: SceneParseResult,
    rule: RulePackItem,
    judgment: ClauseJudgment,
) -> ClauseJudgment:
    scope = scene.observation_scope
    updated = judgment.model_copy(deep=True)

    # 1) 场景明显不匹配
    if updated.applicability == "unmatched":
        updated.verdict = "not_applicable"
        updated.reflection_note = "复核后认为当前图片与该条款场景不匹配，调整为不适用。"
        return updated

    # 2) 可见边界不足时，不允许强判
    if rule.primary_visibility == "内部" and not scope.inside_visible:
        if updated.verdict in {"compliant", "non_compliant"}:
            updated.verdict = "doubtful"
            updated.reflection_note = "该条款需内部可见，但当前图片未提供足够箱内细节，已降级为存疑。"
        if "缺少开箱近景或箱内元件细节" not in updated.missing_evidence:
            updated.missing_evidence.append("缺少开箱近景或箱内元件细节")
        return updated

    if rule.primary_visibility == "台账" and not scope.ledger_available:
        if updated.verdict in {"compliant", "non_compliant"}:
            updated.verdict = "doubtful"
            updated.reflection_note = "该条款依赖台账/记录证据，单张图片不足以形成强结论，已降级为存疑。"
        if "缺少检测记录、台账或系统图等资料证据" not in updated.missing_evidence:
            updated.missing_evidence.append("缺少检测记录、台账或系统图等资料证据")
        return updated

    if (
        (not scope.parameter_readable)
        and updated.verdict in {"compliant", "non_compliant"}
        and any(key in rule.clause_text.lower() for key in ("ip", "30ma", "0.1s"))
    ):
        updated.verdict = "doubtful"
        updated.reflection_note = "该条款涉及参数读取，但当前图片参数不可读，已降级为存疑。"
        if "缺少可读参数近景" not in updated.missing_evidence:
            updated.missing_evidence.append("缺少可读参数近景")
        return updated

    # 3) 缺少直接证据时，不保留违规
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


def _llm_enabled(state: InspectionState) -> bool:
    if state.get("dry_run"):
        return False
    return bool(os.getenv("OPENROUTER_API_KEY"))


def reflect_judgments_node(state: InspectionState) -> InspectionState:
    scene = SceneParseResult.model_validate(state["scene_parse"])
    client = get_llm() if _llm_enabled(state) else None

    final_judgments: list[dict] = []
    for item in state.get("initial_judgments", []):
        judgment = ClauseJudgment.model_validate(item)
        candidate = next(
            (
                cand
                for cand in state.get("candidate_rules", [])
                if cand.get("rulepack_id") == judgment.rulepack_id
            ),
            None,
        )
        rule = RulePackItem.model_validate(candidate) if candidate else RulePackItem.model_validate(
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

        guarded = _apply_guardrails(scene, rule, judgment)

        # 仅对“强结论”做一次 LLM 反思，避免成本过高
        if client is None or guarded.verdict not in {"compliant", "non_compliant"}:
            final_judgments.append(guarded.model_dump())
            continue

        prompt = REFLECTION_PROMPT.format(
            scene_json=json.dumps(scene.model_dump(), ensure_ascii=False, indent=2),
            rule_json=json.dumps(candidate or rule.model_dump(), ensure_ascii=False, indent=2),
            judgment_json=json.dumps(guarded.model_dump(), ensure_ascii=False, indent=2),
        )
        try:
            raw = call_llm(client, prompt, model=TEXR_MODEL, temperature=0.0)
            data = safe_load_json(raw, default={})
            if not data:
                raise ValueError("reflection llm 未返回 JSON")

            revised = guarded.model_dump()
            revised.update(data)
            revised.setdefault("spec_name", guarded.spec_name)
            revised.setdefault("clause_text", guarded.clause_text)
            revised.setdefault("visibility_tag", guarded.visibility_tag)
            revised.setdefault("trigger_name", guarded.trigger_name)
            revised.setdefault("disposal_suggestion", guarded.disposal_suggestion)
            revised.setdefault("retrieval_score", guarded.retrieval_score)
            final = ClauseJudgment.model_validate(revised)
        except Exception:
            final = guarded

        final_judgments.append(final.model_dump())

    state["final_judgments"] = final_judgments

    log_metrics(
        {
            "final_judgments": len(final_judgments),
        }
    )
    return state
