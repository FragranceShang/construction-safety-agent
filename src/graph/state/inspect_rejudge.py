from __future__ import annotations

import json
import os

from inspection.models import (
    ActionObservation,
    ClauseJudgment,
    RulePackItem,
    SceneParseResult,
)
from inspection.prompt import REJUDGE_PROMPT
from model.inspection_state import InspectionState
from utils.inspection_logger import inspection_logger
from utils.json_utils import safe_load_json
from utils.llm import VISION_JUDGE_MODEL, call_multimodal_llm, get_llm
from utils.wandb import log_metrics


_DEFECT_TERMS = ("破损", "缺损", "缺盖", "脱落", "碎裂", "裂纹", "损坏")
_PROTECTION_TERMS = (
    "未封闭开口",
    "开口可见",
    "裸露导线",
    "未见保护措施",
    "无保护措施",
    "受力",
    "尖锐断口",
)
_RAIN_TERMS = ("未见明显防雨", "缺少防雨", "未见防雨")


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
            "judge_boundary": {"compliant": "", "non_compliant": "", "doubtful": ""},
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


def _unique_image_paths(
    observations: list[ActionObservation], fallback_image: str
) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for path in [fallback_image] + [
        img for obs in observations for img in obs.image_paths
    ]:
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _heuristic_rejudge(
    scene: SceneParseResult,
    rule: RulePackItem,
    judgment: ClauseJudgment,
    observations: list[ActionObservation],
) -> ClauseJudgment:
    updated = judgment.model_copy(deep=True)
    observation_text = "\n".join(
        [
            obs.summary
            + "\n"
            + "\n".join(obs.usable_evidence)
            + "\n"
            + "\n".join(obs.extracted_texts)
            for obs in observations
        ]
    )

    if not observations:
        return updated

    if all(obs.status in {"skipped", "no_gain"} for obs in observations):
        for obs in observations:
            if obs.summary and obs.summary not in updated.missing_evidence:
                updated.missing_evidence.append(obs.summary)
        return updated

    if rule.spec_clause == "6.4.1" or any(
        key in rule.clause_text for key in ("完好", "破损", "不合格")
    ):
        if any(term in observation_text for term in _DEFECT_TERMS):
            updated.applicability = "matched"
            updated.verdict = "non_compliant"
            updated.evidence_against = sorted(
                set(
                    updated.evidence_against
                    + [
                        obs
                        for obs in observation_text.splitlines()
                        if any(term in obs for term in _DEFECT_TERMS)
                    ]
                )
            )[:4]
            updated.missing_evidence = [
                item for item in updated.missing_evidence if "破损" not in item
            ]
            updated.reason = (
                "follow-up 局部检查补到了破损/缺损类直接证据，可升级为疑似违规。"
            )
            return updated

    if rule.spec_clause == "6.3.16" or any(
        key in rule.clause_text for key in ("保护措施", "承受外力", "连接器")
    ):
        if any(term in observation_text for term in _PROTECTION_TERMS):
            updated.applicability = "matched"
            updated.verdict = "non_compliant"
            updated.evidence_against = sorted(
                set(
                    updated.evidence_against
                    + [
                        obs
                        for obs in observation_text.splitlines()
                        if any(term in obs for term in _PROTECTION_TERMS)
                    ]
                )
            )[:4]
            updated.reason = "follow-up 局部检查补到了开口异常/线缆保护不足等直接证据，可升级为疑似违规。"
            return updated

    if "防雨" in rule.clause_text and any(
        term in observation_text for term in _RAIN_TERMS
    ):
        updated.applicability = "matched"
        updated.verdict = "non_compliant"
        updated.evidence_against = sorted(
            set(
                updated.evidence_against
                + [
                    obs
                    for obs in observation_text.splitlines()
                    if any(term in obs for term in _RAIN_TERMS)
                ]
            )
        )[:3]
        updated.reason = "follow-up 外观检查补到了户外缺少防雨线索，可升级为疑似违规。"
        return updated

    if any(obs.action_type == "OCR" and obs.extracted_texts for obs in observations):
        if any(key in rule.clause_text for key in ("警示", "标识", "编号", "名称")):
            updated.evidence_for = sorted(
                set(updated.evidence_for + ["follow-up OCR 读取到相关标识/文字内容"])
            )
            updated.reason = (
                updated.reason or "已补到一定文字证据，但仍需结合条款细则复核。"
            )

    if any(obs.usable_evidence for obs in observations):
        updated.missing_evidence = [
            item
            for item in updated.missing_evidence
            if not any(key in item for key in ("特写", "近景", "局部"))
        ]
    return updated


def rejudge_followup_node(state: InspectionState) -> InspectionState:
    inspection_logger.info(
        "rejudge_followup_node | followup_plan=%s", state.get("followup_plan")
    )

    scene = SceneParseResult.model_validate(state["scene_parse"])
    client = get_llm() if _vlm_enabled(state) else None
    candidate_map = {
        item["rulepack_id"]: item for item in state.get("candidate_rules", [])
    }

    obs_map: dict[str, list[ActionObservation]] = {}
    for item in state.get("action_observations", []):
        obs = ActionObservation.model_validate(item)
        obs_map.setdefault(obs.rulepack_id, []).append(obs)

    rejudged: list[dict] = []
    for item in state.get("initial_judgments", []):
        judgment = ClauseJudgment.model_validate(item)
        observations = obs_map.get(judgment.rulepack_id, [])
        candidate = candidate_map.get(judgment.rulepack_id)
        rule = _rule_from_candidate_or_judgment(candidate, judgment)

        if not observations:
            rejudged.append(judgment.model_dump())
            continue

        if client is None:
            revised = _heuristic_rejudge(scene, rule, judgment, observations)
            rejudged.append(revised.model_dump())
            continue

        prompt = REJUDGE_PROMPT.format(
            scene_json=json.dumps(scene.model_dump(), ensure_ascii=False, indent=2),
            rule_json=json.dumps(
                (candidate or rule.model_dump()), ensure_ascii=False, indent=2
            ),
            judgment_json=json.dumps(
                judgment.model_dump(), ensure_ascii=False, indent=2
            ),
            observations_json=json.dumps(
                [obs.model_dump() for obs in observations], ensure_ascii=False, indent=2
            ),
        )
        image_paths = _unique_image_paths(observations, state["image_path"])
        try:
            raw = call_multimodal_llm(
                client,
                image_paths=image_paths,
                instruction=prompt,
                model=VISION_JUDGE_MODEL,
                temperature=0.0,
                max_tokens=1400,
            )
            data = safe_load_json(raw, default={}) or {}
            data.setdefault("rulepack_id", judgment.rulepack_id)
            data.setdefault("spec_clause", judgment.spec_clause)
            data.setdefault("spec_name", judgment.spec_name)
            data.setdefault("clause_text", judgment.clause_text)
            data.setdefault("visibility_tag", judgment.visibility_tag)
            data.setdefault("trigger_name", judgment.trigger_name)
            data.setdefault("disposal_suggestion", judgment.disposal_suggestion)
            data.setdefault("retrieval_score", judgment.retrieval_score)
            data.setdefault("reflection_note", judgment.reflection_note)
            revised = ClauseJudgment.model_validate(data)
        except Exception:
            revised = _heuristic_rejudge(scene, rule, judgment, observations)
        rejudged.append(revised.model_dump())

    state["rejudge_judgments"] = rejudged
    # log_metrics(
    #     {
    #         "rejudge_judgments": len(rejudged),
    #         "rejudge_non_compliant": sum(
    #             1 for item in rejudged if item.get("verdict") == "non_compliant"
    #         ),
    #     }
    # )
    return state
