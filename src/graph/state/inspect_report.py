from __future__ import annotations

import os
from pathlib import Path

from inspection.models import ClauseJudgment, SceneParseResult
from inspection.service import build_markdown_report, build_report_payload
from model.inspection_state import InspectionState
from utils import const
from utils.json_utils import dump_json
from utils.wandb import log_metrics


def generate_report_node(state: InspectionState) -> InspectionState:
    scene = SceneParseResult.model_validate(state["scene_parse"])
    judgments = [ClauseJudgment.model_validate(item) for item in state.get("final_judgments", [])]

    report = build_report_payload(
        image_path=state["image_path"],
        rulepack_path=state.get("rulepack_path", ""),
        question=state.get("question", ""),
        scene_parse=scene,
        judgments=judgments,
    )

    os.makedirs(const.output_dir, exist_ok=True)
    state["report_markdown_path"] = const.inspection_report_md
    state["report_json_path"] = const.inspection_report_json

    markdown = build_markdown_report(report)
    Path(const.inspection_report_md).write_text(markdown, encoding="utf-8")
    dump_json(report.model_dump(), const.inspection_report_json, indent=2)

    state["answer"] = markdown

    log_metrics(
        {
            "violations": report.summary.non_compliant,
            "doubtful": report.summary.doubtful,
            "compliant": report.summary.compliant,
        }
    )
    return state
