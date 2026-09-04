from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_DATA_ROOT,
    MODEL_SETTINGS,
    PROJECT_ROOT,
    REALTIME_VIDEO_FPS,
    TASKS,
)
from data import read_sample, scan_pairs, warn
from evaluation import evaluate_good_bad, evaluate_reps, model_query, parse_json
from realtime import call_model
from results import rebuild_summaries, unique_result_path


def print_dry_run(
    task: str, model_key: str, pairs: list[tuple[Path, Path]]
) -> None:
    print(f"\n=== DRY RUN: {task} ===")
    if not pairs:
        print("No paired videos found.")
    for video, annotation in pairs:
        print(f"\nVideo: {video}\nAnnotation: {annotation}")
        try:
            query, gold = read_sample(annotation)
            print("Query:\n" + model_query(query, task, model_key))
            print("Gold events:\n" + json.dumps(gold, ensure_ascii=False, indent=2))
        except Exception as error:
            warn(f"Could not read {annotation}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FitInteract-Bench MVP.")
    parser.add_argument("--task", choices=(*TASKS, "all"), required=True)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_SETTINGS),
        default="minicpm",
        help="Realtime model backend (default: minicpm)",
    )
    parser.add_argument("--sample", help="Run only this sample directory or video name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument(
        "--realtime-fps",
        type=int,
        default=REALTIME_VIDEO_FPS,
        help="Video frame upload rate (default: 10 FPS)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if not 1 <= args.realtime_fps <= 10:
        parser.error("--realtime-fps must be between 1 and 10")
    return args


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    results_root = args.results_root.resolve()
    if not data_root.is_dir():
        print(f"ERROR: Data root is not a directory: {data_root}", file=sys.stderr)
        return 2

    if args.task == "all":
        tasks = list(TASKS)
    else:
        tasks = [args.task]

    sample = None
    if args.sample:
        sample = args.sample.lower()
    pairs = {
        task: [
            pair
            for pair in scan_pairs(data_root, task)
            if sample is None
            or sample in {pair[0].stem.lower(), pair[0].parent.name.lower()}
        ][: args.limit]
        for task in tasks
    }
    if sample and not any(pairs.values()):
        print(f"ERROR: Sample not found: {args.sample}", file=sys.stderr)
        return 2
    if args.dry_run:
        for task in tasks:
            print_dry_run(task, args.model, pairs[task])
        print("\nDry run complete. No API call was made and no result file was written.")
        return 0

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    settings = MODEL_SETTINGS[args.model]
    api_key_name = settings.get("api_key_env")
    api_key = os.getenv(api_key_name) if api_key_name else None
    if api_key_name and not api_key:
        print(f"ERROR: {api_key_name} is not set.", file=sys.stderr)
        return 2
    try:
        if args.model == "minicpm":
            import websockets  # noqa: F401
        else:
            from dashscope.audio.qwen_omni import OmniRealtimeConversation  # noqa: F401
    except Exception as error:
        print(f"ERROR: Could not import {args.model} client: {error}", file=sys.stderr)
        return 2

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    total = sum(len(items) for items in pairs.values())
    completed = 0
    for task in tasks:
        task_root = results_root / task
        for video_path, annotation_path in pairs[task]:
            completed += 1
            print(f"[{completed}/{total}] {task}: {video_path.name}")
            try:
                source_query, gold = read_sample(annotation_path)
            except Exception as exc:
                warn(f"Could not read {annotation_path}: {type(exc).__name__}: {exc}")
                continue
            query = model_query(source_query, task, args.model)
            raw = ""
            assembled = ""
            parsed = None
            parse_error = None
            error = None
            api_success = False
            input_details: dict[str, Any] = {}

            try:
                raw, assembled, input_details = call_model(
                    api_key,
                    args.model,
                    task,
                    query,
                    video_path,
                    args.realtime_fps,
                )
                if (
                    args.model == "minicpm"
                    and not input_details.get("tts_disabled_verified")
                ):
                    raise RuntimeError(
                        "MiniCPM backend ignored generate_audio=false: "
                        f"kind=audio {input_details.get('kind_audio_count')}, "
                        f"n_tts_tokens {input_details.get('n_tts_tokens')}, "
                        f"cost_tts_ms {input_details.get('cost_tts_ms')}"
                    )
                api_success = True
                parsed, parse_error = parse_json(assembled, task)
            except Exception as exc:
                error = f"API request failed: {type(exc).__name__}: {exc}"
                warn(f"{video_path.name}: {error}")

            metrics: dict[str, Any] = {}
            try:
                if task == "reps":
                    metrics = evaluate_reps(parsed, gold)
                else:
                    metrics = evaluate_good_bad(parsed, gold)
            except Exception as exc:
                message = f"Annotation/evaluation failed: {type(exc).__name__}: {exc}"
                if error:
                    error = f"{error}; {message}"
                else:
                    error = message
                warn(f"{annotation_path.name}: {message}")

            if api_success:
                print(f"Raw response: {raw}")
                print(f"Assembled response: {assembled}")
                print(f"Parse success: {parsed is not None}")
                if parse_error:
                    print(f"Parse error: {parse_error}")
                print("Metrics: " + json.dumps(metrics, ensure_ascii=False))

            result = {
                "run_id": run_id,
                "video_id": video_path.stem,
                "task": task,
                "video_path": str(video_path),
                "annotation_path": str(annotation_path),
                "query": source_query,
                "model_query": query,
                "input_details": input_details,
                "model": settings["model"],
                "base_url": settings["base_url"],
                "api_success": api_success,
                "parse_success": parsed is not None,
                "raw_response": raw,
                "assembled_response": assembled,
                "parsed_response": parsed,
                "parse_error": parse_error,
                "gold": gold,
                "metrics": metrics,
                "error": error,
            }
            result_path = unique_result_path(task_root, video_path.stem, run_id)
            try:
                task_root.mkdir(parents=True, exist_ok=True)
                with result_path.open("x", encoding="utf-8", newline="\n") as handle:
                    json.dump(result, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                rebuild_summaries(results_root)
                print(f"Saved: {result_path}")
            except Exception as exc:
                warn(f"Could not save {video_path.name}: {type(exc).__name__}: {exc}")

            if completed < total and args.delay:
                time.sleep(args.delay)

    print(f"Completed {completed} sample(s). Results root: {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
