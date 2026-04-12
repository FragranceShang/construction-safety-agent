from __future__ import annotations

from .schemas import Sample


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_by_rulepack(samples: list[Sample]) -> list[dict[str, float | int | str | None]]:
    rulepack_ids = sorted(
        set().union(
            *(set(sample.judgments) | set(sample.gold_judgments) for sample in samples)
        )
        if samples
        else set()
    )
    metrics: list[dict[str, float | int | str | None]] = []

    for rulepack_id in rulepack_ids:
        gold_samples = [sample for sample in samples if rulepack_id in sample.gold_judgments]
        pred_samples = [sample for sample in samples if rulepack_id in sample.judgments]
        matched_samples = [
            sample
            for sample in samples
            if rulepack_id in sample.gold_judgments and rulepack_id in sample.judgments
        ]

        support = len(gold_samples)
        predicted_count = len(pred_samples)
        tp = len(matched_samples)
        fp = predicted_count - tp
        fn = support - tp

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        correct_verdict = sum(
            1
            for sample in matched_samples
            if sample.judgments[rulepack_id] == sample.gold_judgments[rulepack_id]
        )

        gold_nc_samples = [
            sample
            for sample in gold_samples
            if sample.gold_judgments[rulepack_id] == "non_compliant"
        ]
        tp_nc = sum(
            1
            for sample in gold_nc_samples
            if sample.judgments.get(rulepack_id) == "non_compliant"
        )
        pred_doubtful = sum(
            1 for sample in pred_samples if sample.judgments[rulepack_id] == "doubtful"
        )

        metrics.append(
            {
                "rulepack_id": rulepack_id,
                "support": support,
                "predicted_count": predicted_count,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "verdict_accuracy": _safe_divide(correct_verdict, len(matched_samples)),
                "non_compliant_recall": (
                    None if len(gold_nc_samples) == 0 else tp_nc / len(gold_nc_samples)
                ),
                "doubtful_rate": _safe_divide(pred_doubtful, predicted_count),
            }
        )

    return metrics
