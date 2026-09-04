import json
import sys
from pathlib import Path
from typing import Any


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def scan_pairs(data_root: Path, task: str) -> list[tuple[Path, Path]]:
    task_dir = data_root / task
    if not task_dir.is_dir():
        warn(f"No data directory found for task '{task}' under {data_root}")
        return []

    pairs = []
    for sample_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
        annotations = list(sample_dir.glob("*.json"))
        if len(annotations) != 1:
            warn(f"Expected one JSON annotation in: {sample_dir}")
            continue
        annotation_path = annotations[0]
        try:
            document = json.loads(annotation_path.read_text(encoding="utf-8"))
            video_path = sample_dir / document["video_path"]
        except (OSError, json.JSONDecodeError, KeyError) as error:
            warn(f"Could not read {annotation_path}: {error}")
            continue
        if not video_path.is_file():
            warn(f"Video not found: {video_path}")
            continue
        pairs.append((video_path, annotation_path))
    return pairs


def read_sample(path: Path) -> tuple[str, list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    events = [
        {
            "start_time_s": float(item["time_window_sec"]["start"]),
            "end_time_s": float(item["time_window_sec"]["end"]),
            "action": "COACH",
            "text": str(item["text"]).strip(),
        }
        for item in document["annotation"]
    ]
    return str(document["query"]).strip(), events
