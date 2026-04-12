from __future__ import annotations

from .schemas import Sample, VERDICT_LABELS


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_overall(samples: list[Sample]) -> dict[str, float | int | None]:
    tp = 0
    fp = 0
    fn = 0
    correct_verdict_total = 0
    matched_total = 0
    aligned_pairs: list[tuple[str, str]] = []
    correct_final_result = 0
    gold_final_violation = 0
    tp_final_violation = 0

    groups = set()
    gold_rulepacks = set()

    for sample in samples:
        pred_ids = set(sample.judgments)
        gold_ids = set(sample.gold_judgments)
        matched_ids = pred_ids & gold_ids

        groups.add(sample.group)
        gold_rulepacks.update(gold_ids)

        tp += len(matched_ids)
        fp += len(pred_ids - gold_ids)
        fn += len(gold_ids - pred_ids)

        for rulepack_id in matched_ids:
            pred_verdict = sample.judgments[rulepack_id]
            gold_verdict = sample.gold_judgments[rulepack_id]
            aligned_pairs.append((gold_verdict, pred_verdict))
            matched_total += 1
            if pred_verdict == gold_verdict:
                correct_verdict_total += 1

        if sample.final_verdict == sample.gold_final_verdict:
            correct_final_result += 1
        if sample.gold_final_verdict == "non_compliant":
            gold_final_violation += 1
            if sample.final_verdict == "non_compliant":
                tp_final_violation += 1

    rule_detection_precision = _safe_divide(tp, tp + fp)
    rule_detection_recall = _safe_divide(tp, tp + fn)
    verdict_accuracy = _safe_divide(correct_verdict_total, matched_total)

    support_by_label = {label: 0 for label in VERDICT_LABELS}
    f1_by_label: dict[str, float] = {}
    for label in VERDICT_LABELS:
        label_tp = sum(1 for gold, pred in aligned_pairs if gold == label and pred == label)
        label_fp = sum(1 for gold, pred in aligned_pairs if gold != label and pred == label)
        label_fn = sum(1 for gold, pred in aligned_pairs if gold == label and pred != label)
        support_by_label[label] = sum(1 for gold, _ in aligned_pairs if gold == label)
        label_precision = _safe_divide(label_tp, label_tp + label_fp)
        label_recall = _safe_divide(label_tp, label_tp + label_fn)
        f1_by_label[label] = _f1(label_precision, label_recall)

    support_total = sum(support_by_label.values())
    verdict_weighted_f1 = (
        0.0
        if support_total == 0
        else sum(support_by_label[label] * f1_by_label[label] for label in VERDICT_LABELS)
        / support_total
    )

    gold_nc = support_by_label["non_compliant"]
    tp_nc = sum(
        1 for gold, pred in aligned_pairs if gold == "non_compliant" and pred == "non_compliant"
    )

    return {
        "sample_count": len(samples),
        "group_count": len(groups),
        "rulepack_count": len(gold_rulepacks),
        "rule_detection_precision": rule_detection_precision,
        "rule_detection_recall": rule_detection_recall,
        "rule_detection_f1": _f1(rule_detection_precision, rule_detection_recall),
        "verdict_accuracy": verdict_accuracy,
        "verdict_weighted_f1": verdict_weighted_f1,
        "non_compliant_recall": None if gold_nc == 0 else tp_nc / gold_nc,
        "final_result_accuracy": _safe_divide(correct_final_result, len(samples)),
        "final_violation_recall_strict": (
            None if gold_final_violation == 0 else tp_final_violation / gold_final_violation
        ),
    }
