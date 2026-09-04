import json
import re
from typing import Any

from config import COUNT_RE


def model_query(query: str, task: str, model_key: str) -> str:
    if model_key == "minicpm":
        if task == "reps":
            behavior = (
                "Whenever one complete repetition has visually finished, immediately output "
                "the next count. Remain silent at all other times. Do not wait for speech."
            )
        else:
            behavior = (
                "When coaching becomes useful, immediately output one correction. "
                "Remain silent at all other times."
            )
    else:
        if task == "reps":
            behavior = (
                "At each response opportunity, output every new count since the previous response, "
                "or WAIT if there is none."
            )
        else:
            behavior = (
                "At each response opportunity, output one correction if coaching is useful, "
                "or WAIT if it is not."
            )
    return f"{query} {behavior}"


def validate_response(value: Any) -> str | None:
    if not isinstance(value, dict):
        return "Parsed JSON must be an object"
    events = value.get("events")
    if not isinstance(events, list):
        return "Parsed JSON must contain an events array"
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict):
            return f"Event {index} must be an object"
        if "events" in event:
            return f"Event {index} contains a nested events array"
        if not str(event.get("action") or "").strip():
            return f"Event {index} is missing action"
    return None


def parse_json(raw: str, task: str) -> tuple[Any | None, str | None]:
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return None, f"JSON parse failed: {error}"

    validation_error = validate_response(parsed)
    if validation_error:
        return None, f"JSON structure invalid: {validation_error}"

    if task == "reps":
        allowed = {"COACH"}
    else:
        allowed = {"WAIT", "COACH"}
    for index, event in enumerate(parsed["events"], 1):
        normalized = str(event["action"]).strip().upper()
        if normalized not in allowed:
            return None, (
                f"JSON structure invalid: action {event['action']!r} in event {index} "
                f"is not allowed for {task}; expected {sorted(allowed)}"
            )
        event["action"] = normalized
    return parsed, None


def prediction_events(parsed: Any) -> list[dict[str, Any]]:
    if parsed:
        return parsed["events"]
    return []


def action(event: dict[str, Any]) -> str:
    return str(event.get("action") or "").strip().upper()


def timestamp(event: dict[str, Any] | None) -> float | None:
    if not event:
        return None
    value = event.get("time_s", event.get("start_time_s"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def in_window(value: float | None, gold: dict[str, Any]) -> bool:
    start, end = gold["start_time_s"], gold["end_time_s"]
    return (
        value is not None
        and start is not None
        and end is not None
        and start <= value <= end
    )


def evaluate_reps(parsed: Any, gold: list[dict[str, Any]]) -> dict[str, Any]:
    gold_events = [event for event in gold if event["action"] == "COACH"]
    if parsed is None:
        return {
            "gold_count_action": "COACH", "gold_rep_count": len(gold_events),
            "pred_rep_count": None, "count_exact": None, "count_absolute_error": None,
            "timing_correct_count": None, "timing_total": len(gold_events),
            "predicted_event_count": None, "timing_matches": [],
        }

    events = prediction_events(parsed)
    pred_events = [event for event in events if action(event) == "COACH"]
    matches = []
    for index, gold_event in enumerate(gold_events):
        predicted = pred_events[index] if index < len(pred_events) else None
        matches.append(
            {
                "index": index + 1,
                "gold_window": [gold_event["start_time_s"], gold_event["end_time_s"]],
                "gold_text": gold_event["text"],
                "predicted_time_s": timestamp(predicted),
                "predicted_text": (
                    predicted.get("text") if predicted else None
                ),
                "correct": in_window(timestamp(predicted), gold_event),
            }
        )
    rep_count = len(pred_events)
    return {
        "gold_count_action": "COACH", "gold_rep_count": len(gold_events),
        "pred_rep_count": rep_count, "count_exact": rep_count == len(gold_events),
        "count_absolute_error": abs(rep_count - len(gold_events)),
        "timing_correct_count": sum(match["correct"] for match in matches),
        "timing_total": len(gold_events), "predicted_event_count": len(events),
        "timing_matches": matches,
    }


def evaluate_good_bad(parsed: Any, gold: list[dict[str, Any]]) -> dict[str, Any]:
    gold_actions = [event["action"] for event in gold]
    if parsed is None:
        return {
            "gold_actions": gold_actions, "pred_actions": [], "actions_exact": None,
            "coach_timing_correct": None, "coach_timing_correct_count": None,
            "coach_timing_total": len([event for event in gold if event["action"] == "COACH"]),
            "predicted_event_count": None, "over_intervention": None,
        }

    events = prediction_events(parsed)
    pred_actions = [action(event) for event in events]
    gold_coach = [event for event in gold if event["action"] == "COACH"]
    pred_coach = [event for event in events if action(event) == "COACH"]
    coach_matches = []
    for index, gold_event in enumerate(gold_coach):
        predicted = pred_coach[index] if index < len(pred_coach) else None
        coach_matches.append(
            {
                "index": index + 1,
                "gold_window": [gold_event["start_time_s"], gold_event["end_time_s"]],
                "gold_text": gold_event["text"],
                "predicted_time_s": timestamp(predicted),
                "predicted_text": (
                    predicted.get("text") if predicted else None
                ),
                "correct": in_window(timestamp(predicted), gold_event),
            }
        )

    wait_windows = [event for event in gold if event["action"] == "WAIT"]
    interventions = [
        {
            "action": action(event),
            "time_s": timestamp(event),
            "text": event.get("text"),
            "wait_window": [wait["start_time_s"], wait["end_time_s"]],
        }
        for event in events
        for wait in wait_windows
        if action(event) == "COACH" and in_window(timestamp(event), wait)
    ]
    correct_count = sum(match["correct"] for match in coach_matches)
    return {
        "gold_actions": gold_actions, "pred_actions": pred_actions,
        "actions_exact": pred_actions == gold_actions,
        "coach_timing_correct": (
            correct_count == len(coach_matches) if coach_matches else None
        ),
        "coach_timing_correct_count": correct_count, "coach_timing_total": len(coach_matches),
        "predicted_event_count": len(events),
        "coach_timing_matches": coach_matches,
        "over_intervention": bool(interventions),
        "over_intervention_events": interventions,
    }


def assemble_reps(stream_events: list[dict[str, Any]]) -> str:
    events = []
    buffer = ""
    for stream_event in stream_events:
        if stream_event["event_type"] != "TextDelta":
            continue
        buffer += stream_event["text"]
        while match := COUNT_RE.search(buffer):
            event = {
                "time_s": stream_event["time_s"],
                "action": "COACH",
                "text": match.group(1) + ".",
            }
            for name in (
                "decision", "decision_index", "phase", "wall_time_s", "input_count"
            ):
                if name in stream_event:
                    event[name] = stream_event[name]
            events.append(event)
            buffer = buffer[match.end():]
    return json.dumps({"rep_count": len(events), "events": events}, ensure_ascii=False)


def assemble_coaching(stream_events: list[dict[str, Any]]) -> str:
    events = []
    buffer = ""
    for stream_event in stream_events:
        if stream_event["event_type"] != "TextDelta":
            if stream_event["event_type"] in {"ResponseDone", "TurnCompleted"}:
                buffer = ""
            continue
        buffer += stream_event["text"]
        while match := re.search(r"[^.!?]+[.!?]", buffer):
            text = match.group().strip()
            buffer = buffer[match.end():]
            if text.rstrip(".!?").strip().upper() == "WAIT":
                continue
            event = {
                "time_s": stream_event["time_s"],
                "action": "COACH",
                "text": text,
            }
            for name in (
                "decision", "decision_index", "phase", "wall_time_s", "input_count"
            ):
                if name in stream_event:
                    event[name] = stream_event[name]
            events.append(event)
    return json.dumps({"events": events}, ensure_ascii=False)
