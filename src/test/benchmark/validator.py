from __future__ import annotations

from typing import Any

from .schemas import Sample, VALID_VERDICTS


REQUIRED_FIELDS = (
    "image_path",
    "group",
    "judgments",
    "final_verdict",
    "gold_judgments",
    "gold_final_verdict",
)


def _require_verdict(value: Any, field_path: str) -> str:
    if value == "not_applicable":
        return "doubtful"
    if value not in VALID_VERDICTS:
        raise ValueError(
            f"{field_path}: invalid verdict {value!r}; expected one of {sorted(VALID_VERDICTS)}"
        )
    return str(value)


def _extract_rule_verdicts(items: Any, field_path: str) -> dict[str, str]:
    if not isinstance(items, list):
        raise ValueError(f"{field_path}: expected a list")

    verdicts: dict[str, str] = {}
    for idx, item in enumerate(items):
        item_path = f"{field_path}[{idx}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_path}: expected an object")
        if "rulepack_id" not in item:
            raise ValueError(f"{item_path}.rulepack_id: missing required field")
        if "verdict" not in item:
            raise ValueError(f"{item_path}.verdict: missing required field")

        rulepack_id = item["rulepack_id"]
        if not isinstance(rulepack_id, str) or not rulepack_id:
            raise ValueError(f"{item_path}.rulepack_id: expected a non-empty string")
        if rulepack_id in verdicts:
            raise ValueError(f"{field_path}: duplicate rulepack_id {rulepack_id!r}")

        verdicts[rulepack_id] = _require_verdict(item["verdict"], f"{item_path}.verdict")
    return verdicts


def validate_records(records: list[dict[str, Any]]) -> list[Sample]:
    seen_image_paths: set[str] = set()
    samples: list[Sample] = []

    for idx, record in enumerate(records):
        sample_path = f"sample[{idx}]"
        for field in REQUIRED_FIELDS:
            if field not in record:
                raise ValueError(f"{sample_path}.{field}: missing required field")

        image_path = record["image_path"]
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"{sample_path}.image_path: expected a non-empty string")
        if image_path in seen_image_paths:
            raise ValueError(f"{sample_path}.image_path: duplicate image_path {image_path!r}")
        seen_image_paths.add(image_path)

        group = record["group"]
        if not isinstance(group, str):
            raise ValueError(f"{sample_path}.group: expected a string")

        samples.append(
            Sample(
                image_path=image_path,
                group=group,
                judgments=_extract_rule_verdicts(record["judgments"], f"{sample_path}.judgments"),
                final_verdict=_require_verdict(record["final_verdict"], f"{sample_path}.final_verdict"),
                gold_judgments=_extract_rule_verdicts(
                    record["gold_judgments"], f"{sample_path}.gold_judgments"
                ),
                gold_final_verdict=_require_verdict(
                    record["gold_final_verdict"], f"{sample_path}.gold_final_verdict"
                ),
                raw=record,
            )
        )

    return samples
