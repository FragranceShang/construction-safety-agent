from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_VERDICTS = {"non_compliant", "doubtful", "compliant"}
VERDICT_LABELS = ("non_compliant", "doubtful", "compliant")


@dataclass(frozen=True)
class Sample:
    image_path: str
    group: str
    judgments: dict[str, str]
    final_verdict: str
    gold_judgments: dict[str, str]
    gold_final_verdict: str
    raw: dict[str, Any]
