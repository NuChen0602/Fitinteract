import csv
import json
from pathlib import Path
from typing import Any

from config import SUMMARY_FIELDS, TASKS


def unique_result_path(directory: Path, video_id: str, run_id: str) -> Path:
    path = directory / f"{video_id}.json"
    if not path.exists():
        return path
    path = directory / f"{video_id}_{run_id}.json"
    counter = 2
    while path.exists():
        path = directory / f"{video_id}_{run_id}_{counter}.json"
        counter += 1
    return path


def make_summary(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    metrics = result["metrics"]
    row = {
        "run_id": result["run_id"], "result_path": str(result_path),
        "video_id": result["video_id"], "task": result["task"],
        "model": result["model"], "api_success": result["api_success"],
        "parse_success": result["parse_success"],
        "error": result["error"] or result["parse_error"] or "",
    }
    for name in SUMMARY_FIELDS:
        if name in metrics:
            value = metrics[name]
            if isinstance(value, list):
                value = json.dumps(value, ensure_ascii=False)
            row[name] = value
    return row


def write_summary(path: Path, result_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result_path in result_paths:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            row = make_summary(result, result_path.resolve())
            writer.writerow({name: row.get(name) for name in SUMMARY_FIELDS})


def rebuild_summaries(results_root: Path) -> None:
    all_results = []
    for task in TASKS:
        task_results = sorted((results_root / task).glob("*.json"))
        if task_results:
            write_summary(results_root / task / "summary.csv", task_results)
            all_results.extend(task_results)
    if all_results:
        write_summary(results_root / "summary.csv", sorted(all_results))
