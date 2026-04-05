from __future__ import annotations

from collections import Counter
from typing import Iterable

from .models import ClauseJudgment, InspectionReport, InspectionSummary, SceneParseResult


VERDICT_LABEL = {
    "non_compliant": "疑似违规",
    "doubtful": "存疑",
    "compliant": "符合",
    "not_applicable": "不适用",
}

VERDICT_ORDER = {
    "non_compliant": 0,
    "doubtful": 1,
    "compliant": 2,
    "not_applicable": 3,
}


def sort_judgments(judgments: Iterable[ClauseJudgment]) -> list[ClauseJudgment]:
    return sorted(
        judgments,
        key=lambda item: (VERDICT_ORDER.get(item.verdict, 9), -item.retrieval_score, item.spec_clause),
    )


def build_summary(judgments: list[ClauseJudgment]) -> InspectionSummary:
    counter = Counter(item.verdict for item in judgments)
    return InspectionSummary(
        total_candidates=len(judgments),
        non_compliant=counter.get("non_compliant", 0),
        doubtful=counter.get("doubtful", 0),
        compliant=counter.get("compliant", 0),
        not_applicable=counter.get("not_applicable", 0),
    )


def build_final_conclusion(summary: InspectionSummary) -> str:
    if summary.non_compliant > 0:
        return (
            f"本次共核验 {summary.total_candidates} 条候选条款，发现 "
            f"{summary.non_compliant} 条疑似违规、{summary.doubtful} 条存疑。"
        )
    if summary.doubtful > 0:
        return (
            f"本次共核验 {summary.total_candidates} 条候选条款，暂未形成明确违规，"
            f"但存在 {summary.doubtful} 条存疑条款，需补充证据后复核。"
        )
    if summary.compliant > 0:
        return f"本次共核验 {summary.total_candidates} 条候选条款，当前已核验条款未发现明确违规。"
    return "当前图片未形成足够的规则判定基础，建议补充更清晰的近景、开箱照或台账。"


def build_report_payload(
    image_path: str,
    rulepack_path: str,
    question: str,
    scene_parse: SceneParseResult,
    judgments: list[ClauseJudgment],
) -> InspectionReport:
    sorted_items = sort_judgments(judgments)
    summary = build_summary(sorted_items)
    conclusion = build_final_conclusion(summary)
    return InspectionReport(
        image_path=image_path,
        rulepack_path=rulepack_path,
        question=question,
        scene_parse=scene_parse,
        summary=summary,
        judgments=sorted_items,
        final_conclusion=conclusion,
    )


def _render_items(title: str, items: list[ClauseJudgment]) -> list[str]:
    if not items:
        return [f"## {title}", "无。", ""]
    lines = [f"## {title}", ""]
    for idx, item in enumerate(items, start=1):
        lines.append(
            f"{idx}. 【{item.spec_clause}】{VERDICT_LABEL.get(item.verdict, item.verdict)}"
        )
        lines.append(f"   - 条款：{item.clause_text}")
        if item.evidence_for:
            lines.append(f"   - 支持证据：{'；'.join(item.evidence_for)}")
        if item.evidence_against:
            lines.append(f"   - 反向证据：{'；'.join(item.evidence_against)}")
        if item.missing_evidence:
            lines.append(f"   - 缺失证据：{'；'.join(item.missing_evidence)}")
        if item.reason:
            lines.append(f"   - 说明：{item.reason}")
        if item.reflection_note:
            lines.append(f"   - 复核：{item.reflection_note}")
        if item.disposal_suggestion:
            lines.append(f"   - 建议：{item.disposal_suggestion}")
        lines.append("")
    return lines


def build_markdown_report(report: InspectionReport) -> str:
    scene = report.scene_parse
    lines: list[str] = [
        "# 施工安全图片检测报告",
        "",
        f"- 图片：`{report.image_path}`",
        f"- 规则包：`{report.rulepack_path}`",
        f"- 问题：{report.question or '请按规则包判断图片中的施工安全问题'}",
        "",
        "## 总结论",
        report.final_conclusion,
        "",
        "## 场景解析摘要",
        f"- 场景类型：{scene.scene_type or '未识别'}",
        f"- 检查目标：{scene.inspection_target or '未识别'}",
        f"- 摘要：{scene.summary or '无'}",
        f"- 可见对象：{'、'.join(scene.visible_objects) if scene.visible_objects else '无'}",
        f"- 环境：{'、'.join(scene.environment) if scene.environment else '无'}",
        f"- 可见条件：{'、'.join(scene.conditions) if scene.conditions else '无'}",
        f"- 潜在风险：{'、'.join(scene.potential_hazards) if scene.potential_hazards else '无'}",
        f"- 不确定点：{'、'.join(scene.uncertain_points) if scene.uncertain_points else '无'}",
        "",
        "## 观察边界",
        (
            f"- outside_visible={scene.observation_scope.outside_visible}, "
            f"inside_visible={scene.observation_scope.inside_visible}, "
            f"door_label_readable={scene.observation_scope.door_label_readable}, "
            f"parameter_readable={scene.observation_scope.parameter_readable}, "
            f"ledger_available={scene.observation_scope.ledger_available}"
        ),
        "",
    ]

    violations = [item for item in report.judgments if item.verdict == "non_compliant"]
    doubtfuls = [item for item in report.judgments if item.verdict == "doubtful"]
    compliants = [item for item in report.judgments if item.verdict == "compliant"]
    not_applicable = [item for item in report.judgments if item.verdict == "not_applicable"]

    lines.extend(_render_items("疑似违规条款", violations))
    lines.extend(_render_items("存疑条款", doubtfuls))
    lines.extend(_render_items("符合条款", compliants))
    lines.extend(_render_items("不适用条款", not_applicable))

    return "\n".join(lines).strip() + "\n"
