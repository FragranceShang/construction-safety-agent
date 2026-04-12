from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_metrics(output_dir: str | Path, overall: dict[str, Any], by_rulepack: list[dict[str, Any]]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics_overall.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_path / "metrics_by_rulepack.json").write_text(
        json.dumps(by_rulepack, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
