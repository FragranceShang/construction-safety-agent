from __future__ import annotations

import os

from inspection.models import RulePackItem, SceneParseResult
from inspection.retriever import select_candidate_rules_hybrid
from model.inspection_state import InspectionState
from utils.inspection_logger import inspection_logger
from utils.llm import get_llm
from utils.wandb import log_metrics


def _vlm_enabled(state: InspectionState) -> bool:
    if state.get("dry_run"):
        return False
    return bool(os.getenv("OPENROUTER_API_KEY"))


def retrieve_candidate_rules_node(state: InspectionState) -> InspectionState:
    inspection_logger.info(
        "retrieve_candidate_rules_node | question=%s, image_path=%s, rulepack_items=%d",
        state.get("question"),
        state.get("image_path"),
        len(state.get("rulepack_items", [])),
    )

    scene = SceneParseResult.model_validate(state["scene_parse"])
    rules = [
        RulePackItem.model_validate(item) for item in state.get("rulepack_items", [])
    ]

    client = get_llm() if _vlm_enabled(state) else None
    symbolic_candidates, vlm_candidates, candidates = select_candidate_rules_hybrid(
        rules=rules,
        scene=scene,
        question=state.get("question", ""),
        image_path=state.get("image_path", ""),
        client=client,
        top_k_symbolic=5,
        top_k_vlm=3,
        vlm_pool_k=12,
    )

    state["symbolic_candidates"] = symbolic_candidates
    state["vlm_candidates"] = vlm_candidates
    state["candidate_rules"] = candidates

    # log_metrics(
    #     {
    #         "symbolic_candidates": len(symbolic_candidates),
    #         "vlm_candidates": len(vlm_candidates),
    #         "candidate_rules": len(candidates),
    #         "top_candidate_score": (
    #             candidates[0]["retrieval_score"] if candidates else 0
    #         ),
    #     }
    # )
    return state
