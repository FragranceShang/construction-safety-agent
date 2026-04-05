from typing import Any, Dict, List, Optional, TypedDict


class InspectionState(TypedDict, total=False):
    image_path: str
    question: str
    rulepack_path: str
    scene_parse_path: str
    dry_run: bool

    vision_text: str
    scene_parse: Dict[str, Any]
    rulepack_items: List[Dict[str, Any]]
    candidate_rules: List[Dict[str, Any]]
    initial_judgments: List[Dict[str, Any]]
    final_judgments: List[Dict[str, Any]]
    answer: str

    report_markdown_path: str
    report_json_path: str
