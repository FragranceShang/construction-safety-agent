from __future__ import annotations

from inspection.loader import load_rulepack, resolve_rulepack_path, rulepack_to_dicts
from model.inspection_state import InspectionState
from utils.wandb import log_metrics


def load_rulepack_node(state: InspectionState) -> InspectionState:
    rulepack_path = resolve_rulepack_path(project_root=".", explicit_path=state.get("rulepack_path"))
    rulepack = load_rulepack(project_root=".", explicit_path=rulepack_path)

    state["rulepack_path"] = str(rulepack_path)
    state["rulepack_items"] = rulepack_to_dicts(rulepack)

    log_metrics(
        {
            "rulepack_items": len(rulepack),
            "rulepack_path": str(rulepack_path),
        }
    )
    return state
