from __future__ import annotations

import os
from pathlib import Path

from inspection.models import (
    ActionObservation,
    ClauseJudgment,
    FollowupActionPlan,
    SceneParseResult,
)
from inspection.service import build_markdown_report, build_report_payload
from model.inspection_state import InspectionState
from utils import const
from utils.inspection_logger import inspection_logger
from utils.json_utils import dump_json
from utils.wandb import log_metrics


def generate_report_node(state: InspectionState) -> InspectionState:
    inspection_logger.info(
        "generate_report_node | followup_plan=%s", state.get("followup_plan")
    )

    scene = SceneParseResult.model_validate(state["scene_parse"])
    judgments = [
        ClauseJudgment.model_validate(item) for item in state.get("final_judgments", [])
    ]
    followup_plan = [
        FollowupActionPlan.model_validate(item)
        for item in state.get("followup_plan", [])
    ]
    action_observations = [
        ActionObservation.model_validate(item)
        for item in state.get("action_observations", [])
    ]

    report = build_report_payload(
        image_path=state["image_path"],
        rulepack_path=state.get("rulepack_path", ""),
        question=state.get("question", ""),
        scene_parse=scene,
        judgments=judgments,
        symbolic_candidates=state.get("symbolic_candidates", []),
        vlm_candidates=state.get("vlm_candidates", []),
        followup_plan=followup_plan,
        action_observations=action_observations,
    )

    os.makedirs(const.output_dir, exist_ok=True)
    state["report_markdown_path"] = const.inspection_report_md
    state["report_json_path"] = const.inspection_report_json

    markdown = build_markdown_report(report)
    Path(const.inspection_report_md).write_text(markdown, encoding="utf-8")
    dump_json(report.model_dump(), const.inspection_report_json, indent=2)

    state["answer"] = markdown
    rejudged = state.get("rejudge_judgments", [])
    plans = state.get("followup_plan", [])
    candidates = state.get("candidate_rules", [])
    log_metrics(
        {
            "violations": report.summary.non_compliant,
            "doubtful": report.summary.doubtful,
            "compliant": report.summary.compliant,
            "top_candidate_score": (
                candidates[0]["retrieval_score"] if candidates else 0
            ),
            "rejudge_judgments": len(rejudged),
            "rejudge_non_compliant": sum(
                1 for item in rejudged if item.get("verdict") == "non_compliant"
            ),
            "same_image_recoverable": sum(
                1
                for item in plans
                if item.get("observability") == "same_image_recoverable"
            ),
            "action_observations": len(action_observations),
            "executed_actions": sum(
                1 for item in action_observations if item.get("status") == "completed"
            ),
        }
    )
    return state
