from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.test.benchmark.evaluator_overall import evaluate_overall
    from src.test.benchmark.evaluator_rulepack import evaluate_by_rulepack
    from src.test.benchmark.loader import load_jsonl
    from src.test.benchmark.validator import validate_records
    from src.test.benchmark.writer import write_metrics
else:
    from .evaluator_overall import evaluate_overall
    from .evaluator_rulepack import evaluate_by_rulepack
    from .loader import load_jsonl
    from .validator import validate_records
    from .writer import write_metrics


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BENCHMARK_DIR.parent / "input_test_inspection_reports.jsonl"


def run_benchmark(input_path: str | Path = DEFAULT_INPUT, output_dir: str | Path = BENCHMARK_DIR) -> dict[str, object]:
    records = load_jsonl(input_path)
    samples = validate_records(records)
    overall = evaluate_overall(samples)
    by_rulepack = evaluate_by_rulepack(samples)
    write_metrics(output_dir, overall, by_rulepack)
    return {
        "input_path": str(Path(input_path)),
        "output_dir": str(Path(output_dir)),
        "sample_count": overall["sample_count"],
        "rulepack_rows": len(by_rulepack),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inspection benchmark metrics.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to input_test_inspection_reports.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BENCHMARK_DIR),
        help="Directory for metrics_overall.json and metrics_by_rulepack.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(args.input, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
