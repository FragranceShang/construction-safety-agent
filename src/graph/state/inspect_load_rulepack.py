from __future__ import annotations

from inspection.loader import load_rulepack, resolve_rulepack_path, rulepack_to_dicts
from model.inspection_state import InspectionState
from utils.inspection_logger import log_node_end, log_node_info, log_node_start
from utils.wandb import log_metrics


def load_rulepack_node(state: InspectionState) -> InspectionState:
    log_node_start("load_rulepack", explicit_path=state.get("rulepack_path", ""))
    rulepack_path = resolve_rulepack_path(project_root=".", explicit_path=state.get("rulepack_path"))
    rulepack = load_rulepack(project_root=".", explicit_path=rulepack_path)

    state["rulepack_path"] = str(rulepack_path)
    state["rulepack_items"] = rulepack_to_dicts(rulepack)
    preview_clauses = [item.spec_clause for item in rulepack[:3]]
    log_node_info("load_rulepack", "rulepack loaded", resolved_path=str(rulepack_path))
    log_node_end(
        "load_rulepack",
        rule_count=len(rulepack),
        preview_clauses=preview_clauses,
    )

    log_metrics(
        {
            "rulepack_items": len(rulepack),
            "rulepack_path": str(rulepack_path),
        }
    )
    return state
