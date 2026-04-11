from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from main import DEFAULT_QUESTION, run_batch_inspection  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _collect_input_test_images(input_test_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_test_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _write_reports_jsonl(report_paths: Iterable[Path], jsonl_path: Path) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with jsonl_path.open("w", encoding="utf-8") as writer:
        for report_path in sorted(report_paths):
            data = json.loads(report_path.read_text(encoding="utf-8"))
            writer.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def run_input_test_inspection_to_jsonl(
    *,
    dry_run: bool = False,
    question: str = DEFAULT_QUESTION,
    jsonl_path: str | Path | None = None,
) -> dict:
    """
    递归 inspection input/test 下的图片，并将 outputs/test 下的报告 JSON 汇总为 JSONL。
    """
    input_test_dir = PROJECT_ROOT / "input" / "test"
    output_test_dir = PROJECT_ROOT / "outputs" / "test"
    jsonl_output = (
        Path(jsonl_path)
        if jsonl_path
        else Path(__file__).with_name("input_test_inspection_reports.jsonl")
    )
    questions = [question] * len(_collect_input_test_images(input_test_dir))
    image_paths = _collect_input_test_images(input_test_dir)
    results = run_batch_inspection(
        image_paths=[str(path) for path in image_paths],
        question=questions,
        rulepack=str(PROJECT_ROOT / "rulepack.json"),
        dry_run=dry_run,
        print_report=False,
    )

    report_paths = output_test_dir.rglob("*.json")
    report_count = _write_reports_jsonl(report_paths, jsonl_output)
    return {
        "images": len(image_paths),
        "results": len(results),
        "reports": report_count,
        "jsonl_path": str(jsonl_output),
    }


if __name__ == "__main__":
    summary = run_input_test_inspection_to_jsonl()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
