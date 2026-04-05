from __future__ import annotations

import json
from pathlib import Path

from inspection.models import SceneParseResult
from inspection.prompt import VISION_PARSE_JSON_PROMPT
from model.inspection_state import InspectionState
from utils.inspection_logger import log_node_end, log_node_info, log_node_start
from utils.json_utils import safe_load_json
from utils.llm import call_vision_llm, get_llm
from utils.wandb import log_metrics


def inspect_parse_vision_node(state: InspectionState) -> InspectionState:
    """
    1. 优先使用外部提供的 scene_parse_path / scene_parse
    2. 否则调用视觉模型得到结构化 JSON
    """
    log_node_start(
        "parse_vision",
        image_path=state.get("image_path", ""),
        scene_parse_path=state.get("scene_parse_path", ""),
        has_scene_parse=bool(state.get("scene_parse")),
    )
    if state.get("scene_parse"):
        scene = SceneParseResult.model_validate(state["scene_parse"])
        state["scene_parse"] = scene.model_dump()
        state["vision_text"] = json.dumps(state["scene_parse"], ensure_ascii=False, indent=2)
        log_node_info("parse_vision", "using preloaded scene_parse from state")
        log_node_end(
            "parse_vision",
            source="state.scene_parse",
            scene_type=scene.scene_type,
            hazards=len(scene.potential_hazards),
        )
        return state

    scene_parse_path = state.get("scene_parse_path")
    if scene_parse_path:
        scene = SceneParseResult.model_validate_json(
            Path(scene_parse_path).read_text(encoding="utf-8")
        )
        state["scene_parse"] = scene.model_dump()
        state["vision_text"] = json.dumps(state["scene_parse"], ensure_ascii=False, indent=2)
        log_node_info("parse_vision", "loaded scene_parse from json file", scene_parse_path=scene_parse_path)
        log_node_end(
            "parse_vision",
            source="scene_parse_path",
            scene_type=scene.scene_type,
            hazards=len(scene.potential_hazards),
        )
        return state

    log_node_info("parse_vision", "calling vision model for structured scene parsing")
    client = get_llm()
    raw = call_vision_llm(client, state["image_path"], VISION_PARSE_JSON_PROMPT)
    data = safe_load_json(raw, default={})
    if not data:
        log_node_info("parse_vision", "vision model did not return valid json, falling back to summary payload")
        # 极端情况下保底
        data = {
            "scene_type": "未知施工现场图片",
            "inspection_target": "",
            "summary": raw.strip(),
            "visible_objects": [],
            "visible_texts": [],
            "environment": [],
            "conditions": [],
            "potential_hazards": [],
            "uncertain_points": ["视觉模型未返回标准 JSON，已降级为文本摘要"],
            "observation_scope": {
                "outside_visible": True,
                "inside_visible": False,
                "door_label_readable": False,
                "parameter_readable": False,
                "ledger_available": False,
            },
        }

    scene = SceneParseResult.model_validate(data)
    state["scene_parse"] = scene.model_dump()
    state["vision_text"] = json.dumps(state["scene_parse"], ensure_ascii=False, indent=2)
    log_node_end(
        "parse_vision",
        source="vision_model",
        scene_type=scene.scene_type,
        hazards=len(scene.potential_hazards),
        vision_text_length=len(state["vision_text"]),
    )

    log_metrics(
        {
            "vision_text_length": len(state["vision_text"]),
            "image_path": state["image_path"],
        }
    )
    return state
