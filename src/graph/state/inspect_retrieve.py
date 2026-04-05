from __future__ import annotations

from inspection.models import RulePackItem, SceneParseResult
from inspection.retriever import select_candidate_rules
from model.inspection_state import InspectionState
from utils.wandb import log_metrics


def retrieve_candidate_rules_node(state: InspectionState) -> InspectionState:
    scene = SceneParseResult.model_validate(state["scene_parse"])
    rules = [RulePackItem.model_validate(item) for item in state.get("rulepack_items", [])]

    candidates = select_candidate_rules(
        rules=rules,
        scene=scene,
        question=state.get("question", ""),
        top_k_rules=8,
        top_k_triggers=3,
    )
    state["candidate_rules"] = candidates

    log_metrics(
        {
            "candidate_rules": len(candidates),
            "top_candidate_score": candidates[0]["retrieval_score"] if candidates else 0,
        }
    )
    return state
