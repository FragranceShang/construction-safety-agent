from __future__ import annotations

from inspection.models import RulePackItem, SceneParseResult
from inspection.retriever import select_candidate_rules
from model.inspection_state import InspectionState
from utils.inspection_logger import log_node_end, log_node_info, log_node_start
from utils.wandb import log_metrics


def retrieve_candidate_rules_node(state: InspectionState) -> InspectionState:
    scene = SceneParseResult.model_validate(state["scene_parse"])
    rules = [RulePackItem.model_validate(item) for item in state.get("rulepack_items", [])]
    log_node_start(
        "retrieve_candidates",
        rule_count=len(rules),
        scene_type=scene.scene_type,
        question=state.get("question", ""),
    )

    candidates = select_candidate_rules(
        rules=rules,
        scene=scene,
        question=state.get("question", ""),
        top_k_rules=8,
        top_k_triggers=3,
    )
    state["candidate_rules"] = candidates
    top_candidates = [
        f'{item.get("spec_clause", "?")}:{item.get("retrieval_score", 0):.2f}'
        for item in candidates[:5]
    ]
    log_node_info("retrieve_candidates", "candidate retrieval completed", top_candidates=top_candidates)
    log_node_end(
        "retrieve_candidates",
        candidate_count=len(candidates),
        top_score=candidates[0]["retrieval_score"] if candidates else 0,
    )

    log_metrics(
        {
            "candidate_rules": len(candidates),
            "top_candidate_score": candidates[0]["retrieval_score"] if candidates else 0,
        }
    )
    return state
