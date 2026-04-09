from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Verdict = Literal["compliant", "non_compliant", "doubtful", "not_applicable"]
Applicability = Literal["matched", "uncertain", "unmatched"]
ActionType = Literal["OCR", "VISUAL_DETAIL", "GEOMETRY", "VISUAL_CHECK"]
Observability = Literal[
    "same_image_recoverable",
    "needs_new_view",
    "needs_document",
    "not_worth_retry",
]
ActionStatus = Literal["planned", "completed", "skipped", "failed", "no_gain"]


class Trigger(BaseModel):
    trigger_id: str
    trigger_name: str
    visibility_tag: str
    pull_all_clauses_note: Optional[str] = None


class JudgeBoundary(BaseModel):
    compliant: str
    non_compliant: str
    doubtful: str


class RulePackItem(BaseModel):
    rulepack_id: str
    spec_code: str
    spec_clause: str
    spec_name: str
    clause_text: str
    judge_dimension: str
    judge_boundary: JudgeBoundary
    keywords: List[str] = Field(default_factory=list)
    similar_words: List[str] = Field(default_factory=list)
    triggers: List[Trigger] = Field(default_factory=list)
    disposal_suggestion: str = ""

    @property
    def primary_visibility(self) -> str:
        if not self.triggers:
            return "外观"
        return self.triggers[0].visibility_tag

    @property
    def primary_trigger_name(self) -> str:
        if not self.triggers:
            return self.rulepack_id
        return self.triggers[0].trigger_name


class ObservationScope(BaseModel):
    outside_visible: bool = True
    inside_visible: bool = False
    door_label_readable: bool = False
    parameter_readable: bool = False
    ledger_available: bool = False


class SceneParseResult(BaseModel):
    scene_type: str = ""
    inspection_target: str = ""
    summary: str = ""
    visible_objects: List[str] = Field(default_factory=list)
    visible_texts: List[str] = Field(default_factory=list)
    environment: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    potential_hazards: List[str] = Field(default_factory=list)
    uncertain_points: List[str] = Field(default_factory=list)
    observation_scope: ObservationScope = Field(default_factory=ObservationScope)

    def to_search_text(self) -> str:
        parts = [
            self.scene_type,
            self.inspection_target,
            self.summary,
            " ".join(self.visible_objects),
            " ".join(self.visible_texts),
            " ".join(self.environment),
            " ".join(self.conditions),
            " ".join(self.potential_hazards),
            " ".join(self.uncertain_points),
        ]
        scope = self.observation_scope
        parts.append(
            " ".join(
                [
                    "外观可见" if scope.outside_visible else "",
                    "内部可见" if scope.inside_visible else "内部不可见",
                    "门体标识可读" if scope.door_label_readable else "门体标识不可读",
                    "参数可读" if scope.parameter_readable else "参数不可读",
                    "有台账" if scope.ledger_available else "无台账",
                ]
            )
        )
        return "\n".join([part for part in parts if part]).strip()


class ClauseJudgment(BaseModel):
    rulepack_id: str
    spec_clause: str
    spec_name: str
    clause_text: str
    visibility_tag: str
    trigger_name: str
    applicability: Applicability = "uncertain"
    verdict: Verdict = "doubtful"
    evidence_for: List[str] = Field(default_factory=list)
    evidence_against: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    reason: str = ""
    disposal_suggestion: str = ""
    retrieval_score: float = 0.0
    reflection_note: str = ""


class FollowupActionPlan(BaseModel):
    action_id: str
    rulepack_id: str
    spec_clause: str
    action_type: ActionType
    observability: Observability = "same_image_recoverable"
    target: str = ""
    why: str = ""
    expected: str = ""
    roi_request: str = ""
    stop_if: str = ""
    priority: int = 1


class RoiRegion(BaseModel):
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    reason: str = ""


class ActionObservation(BaseModel):
    action_id: str
    rulepack_id: str
    spec_clause: str
    action_type: ActionType
    observability: Observability
    status: ActionStatus = "completed"
    target: str = ""
    roi_regions: List[RoiRegion] = Field(default_factory=list)
    image_paths: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    extracted_texts: List[str] = Field(default_factory=list)
    usable_evidence: List[str] = Field(default_factory=list)
    unresolved: List[str] = Field(default_factory=list)
    summary: str = ""


class InspectionSummary(BaseModel):
    total_candidates: int = 0
    non_compliant: int = 0
    doubtful: int = 0
    compliant: int = 0
    not_applicable: int = 0


class InspectionReport(BaseModel):
    image_path: str
    rulepack_path: str
    question: str = ""
    scene_parse: SceneParseResult
    summary: InspectionSummary
    judgments: List[ClauseJudgment] = Field(default_factory=list)
    final_conclusion: str = ""

    symbolic_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    vlm_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    followup_plan: List[FollowupActionPlan] = Field(default_factory=list)
    action_observations: List[ActionObservation] = Field(default_factory=list)
