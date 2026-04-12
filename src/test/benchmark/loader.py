from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{jsonl_path}:{line_no}: each JSONL line must be an object")
            records.append(record)
    return records
