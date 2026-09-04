import asyncio
import base64
import contextlib
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from config import (
    MINICPM_INFERENCE_CHUNK_SECONDS,
    MODEL_SETTINGS,
    QWEN_TRIGGER_INTERVAL_SECONDS,
    QWEN_VOICE,
    REALTIME_FRAME_WIDTH,
    RESPONSE_TIMEOUT_SECONDS,
)
from evaluation import assemble_coaching, assemble_reps


def extract_realtime_frames(video_path: Path, frame_dir: Path, frame_fps: int) -> list[Path]:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-vf", f"fps={frame_fps},scale={REALTIME_FRAME_WIDTH}:-2", "-q:v", "5",
            str(frame_dir / "frame_%04d.jpg"),
        ],
        check=True,
    )
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg extracted no video frames")
    return frames


def minicpm_realtime_video(
    query: str, frames: list[Path], frame_fps: int
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    import websockets

    async def run() -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        settings = MODEL_SETTINGS["minicpm"]
        frames_per_chunk = frame_fps * MINICPM_INFERENCE_CHUNK_SECONDS
        silent_audio = base64.b64encode(
            bytes(16000 * MINICPM_INFERENCE_CHUNK_SECONDS * 4)
        ).decode("ascii")
        input_count = 0
        phase = "queue"
        expected_decisions: int | None = None
        completed_decisions: set[int] = set()
        drain_complete = asyncio.Event()
        started = time.monotonic()
        text_parts: list[str] = []
        errors: list[str] = []
        stream_events: list[dict[str, Any]] = []
        provider_events: list[dict[str, Any]] = []

        def event_decision_index(event: dict[str, Any]) -> int | None:
            input_id = str(event.get("input_id") or "")
            if input_id.startswith("decision_"):
                try:
                    return int(input_id.removeprefix("decision_"))
                except ValueError:
                    pass
            chunk_index = event.get("chunk_index")
            if isinstance(chunk_index, int):
                return chunk_index + 1
            return None

        def record_event(event: dict[str, Any]) -> str:
            event_type = str(event.get("type") or "")
            decision_index = event_decision_index(event)
            record = {
                **event,
                "phase": phase,
                "wall_time_s": round(time.monotonic() - started, 3),
            }
            if decision_index is not None:
                record.update(
                    decision=f"decision #{decision_index}",
                    decision_index=decision_index,
                    decision_time_s=decision_index * MINICPM_INFERENCE_CHUNK_SECONDS,
                )
            if event_type == "response.output.delta" and event.get("kind") == "audio":
                audio = str(record.pop("audio", ""))
                record["audio_bytes"] = len(base64.b64decode(audio)) if audio else 0
            provider_events.append(record)

            if event_type == "response.output.delta":
                if decision_index is None:
                    raise RuntimeError(
                        "MiniCPM response.output.delta has no decision input_id or chunk_index"
                    )
                completed_decisions.add(decision_index)
                if (
                    expected_decisions is not None
                    and len(completed_decisions) >= expected_decisions
                ):
                    drain_complete.set()
                kind = event.get("kind")
                if kind == "text":
                    text = str(event.get("text") or "")
                    text_parts.append(text)
                    stream_events.append({
                        "event_type": "TextDelta",
                        "phase": phase,
                        "decision": f"decision #{decision_index}",
                        "decision_index": decision_index,
                        "time_s": decision_index * MINICPM_INFERENCE_CHUNK_SECONDS,
                        "wall_time_s": record["wall_time_s"],
                        "input_count": decision_index,
                        "text": text,
                        "end_of_turn": False,
                    })
                elif kind == "listen":
                    stream_events.append({
                        "event_type": "TurnCompleted",
                        "phase": phase,
                        "decision": f"decision #{decision_index}",
                        "decision_index": decision_index,
                        "time_s": decision_index * MINICPM_INFERENCE_CHUNK_SECONDS,
                        "wall_time_s": record["wall_time_s"],
                        "input_count": decision_index,
                    })
            elif event_type == "error":
                error = event.get("error")
                if isinstance(error, dict):
                    errors.append(f"{error.get('code', 'error')}: {error.get('message', error)}")
                else:
                    errors.append(str(error or event))
            return event_type

        async def receive_until(ws: Any, wanted: str) -> None:
            while True:
                raw = await asyncio.wait_for(ws.recv(), RESPONSE_TIMEOUT_SECONDS)
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise RuntimeError("MiniCPM server event must be a JSON object")
                event_type = record_event(event)
                if event_type == "error":
                    raise RuntimeError(f"MiniCPM realtime error: {'; '.join(errors)}")
                if event_type in {wanted, wanted.removeprefix("session.")}:
                    return

        async def receive(ws: Any) -> None:
            async for raw in ws:
                event = json.loads(raw)
                if not isinstance(event, dict):
                    errors.append("MiniCPM server event must be a JSON object")
                    return
                if record_event(event) in {"error", "session.closed"}:
                    return

        async with websockets.connect(
            settings["realtime_url"],
            open_timeout=RESPONSE_TIMEOUT_SECONDS,
            max_size=None,
        ) as ws:
            await receive_until(ws, "session.queue_done")
            phase = "initializing"
            await ws.send(json.dumps({
                "type": "session.init",
                "payload": {
                    "system_prompt": query,
                    "use_tts": False,
                    "config": {
                        "generate_audio": False,
                        "force_listen_count": 0,
                    },
                },
            }))
            await receive_until(ws, "session.created")

            receiver = asyncio.create_task(receive(ws))
            try:
                phase = "video"
                for start in range(0, len(frames), frames_per_chunk):
                    frame_chunk = frames[start:start + frames_per_chunk]
                    input_count += 1
                    await ws.send(json.dumps({
                        "type": "input.append",
                        "input": {
                            "input_id": f"decision_{input_count}",
                            "audio": silent_audio,
                            "video_frames": [
                                base64.b64encode(frame.read_bytes()).decode("ascii")
                                for frame in frame_chunk
                            ],
                            "force_listen": False,
                            "max_slice_nums": 1,
                        },
                    }))
                    await asyncio.sleep(MINICPM_INFERENCE_CHUNK_SECONDS)

                phase = "drain"
                expected_decisions = input_count
                if len(completed_decisions) >= expected_decisions:
                    drain_complete.set()

                drain_waiter = asyncio.create_task(drain_complete.wait())
                done, _pending = await asyncio.wait(
                    {drain_waiter, receiver},
                    timeout=RESPONSE_TIMEOUT_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    drain_waiter.cancel()
                    raise TimeoutError(
                        "Timed out draining MiniCPM inference: "
                        f"{len(completed_decisions)}/{expected_decisions} decisions completed"
                    )
                if receiver in done and not drain_complete.is_set():
                    drain_waiter.cancel()
                    receiver.result()
                    raise RuntimeError(
                        "MiniCPM connection closed during drain: "
                        f"{len(completed_decisions)}/{expected_decisions} decisions completed"
                    )
                await drain_waiter

                phase = "closing"
                await ws.send(json.dumps({"type": "session.close", "reason": "user_stop"}))
                await asyncio.wait_for(receiver, RESPONSE_TIMEOUT_SECONDS)
            finally:
                if not receiver.done():
                    receiver.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await receiver

        if errors:
            raise RuntimeError(f"MiniCPM realtime error: {'; '.join(errors)}")
        return "".join(text_parts).strip(), stream_events, provider_events

    return asyncio.run(run())


def summarize_minicpm_output(
    provider_events: list[dict[str, Any]],
) -> dict[str, Any]:
    audio_output_chunks = 0
    decision_metrics: dict[int, dict[str, Any]] = {}
    decision_indices: set[int] = set()

    for event in provider_events:
        if event.get("type") != "response.output.delta":
            continue
        decision_index = event.get("decision_index")
        if isinstance(decision_index, int):
            decision_indices.add(decision_index)
            metrics = event.get("metrics")
            if isinstance(metrics, dict):
                decision_metrics.setdefault(decision_index, metrics)
        if event.get("kind") == "audio":
            audio_output_chunks += 1

    def total_metric(name: str) -> int | float | None:
        if not decision_indices or set(decision_metrics) != decision_indices:
            return None
        values = [decision_metrics[index].get(name, 0) for index in decision_indices]
        if any(not isinstance(value, (int, float)) for value in values):
            return None
        return sum(values)

    n_tts_tokens = total_metric("n_tts_tokens")
    cost_tts_ms = total_metric("cost_tts_ms")
    return {
        "kind_audio_count": audio_output_chunks,
        "audio_output_chunks": audio_output_chunks,
        "n_tts_tokens": n_tts_tokens,
        "cost_tts_ms": cost_tts_ms,
        "tts_disabled_verified": (
            audio_output_chunks == 0
            and n_tts_tokens == 0
            and cost_tts_ms == 0
        ),
    }


def qwen_realtime_video(
    api_key: str, query: str, frames: list[Path], frame_fps: int
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    from dashscope.audio.qwen_omni import (
        MultiModality,
        OmniRealtimeCallback,
        OmniRealtimeConversation,
    )

    settings = MODEL_SETTINGS["qwen"]
    playback_time = 0.0
    input_count = 0
    started = time.monotonic()
    response_done = threading.Event()
    response_done.set()
    text_parts = []
    errors = []
    stream_events = []
    provider_events = []

    class Callback(OmniRealtimeCallback):
        def on_event(self, response: dict[str, Any]) -> None:
            record = dict(response)
            record["client_time_s"] = round(playback_time, 2)
            record["client_wall_time_s"] = round(time.monotonic() - started, 3)
            record["input_count"] = input_count
            provider_events.append(record)
            event_type = response.get("type")
            if event_type == "response.text.delta":
                text = str(response.get("delta") or "")
                text_parts.append(text)
                stream_events.append({
                    "event_type": "TextDelta", "time_s": record["client_time_s"],
                    "wall_time_s": record["client_wall_time_s"], "input_count": input_count,
                    "text": text, "end_of_turn": False,
                })
            elif event_type == "response.done":
                stream_events.append({
                    "event_type": "ResponseDone", "time_s": record["client_time_s"],
                    "wall_time_s": record["client_wall_time_s"], "input_count": input_count,
                })
                response_done.set()
            elif event_type == "error":
                errors.append(json.dumps(response.get("error", response), ensure_ascii=False))
                response_done.set()

        def on_close(self, code: int, message: str) -> None:
            if not response_done.is_set():
                errors.append(f"connection closed before response.done: {code} {message}")
            response_done.set()

    conversation = OmniRealtimeConversation(
        model=settings["model"],
        callback=Callback(),
        url=settings["base_url"],
        api_key=api_key,
    )
    frame_period = 1 / frame_fps
    silent_audio = bytes(round(16000 * frame_period) * 2)
    next_trigger = QWEN_TRIGGER_INTERVAL_SECONDS
    pending_frames = 0

    def create_response() -> None:
        nonlocal pending_frames
        response_done.clear()
        conversation.commit()
        conversation.create_response(output_modalities=[MultiModality.TEXT])
        pending_frames = 0
        if not response_done.wait(RESPONSE_TIMEOUT_SECONDS):
            raise TimeoutError("Timed out waiting for Qwen response.done")
        if errors:
            raise RuntimeError(f"Qwen realtime error: {'; '.join(errors)}")

    conversation.connect()
    try:
        conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            voice=QWEN_VOICE,
            instructions=query,
            enable_input_audio_transcription=False,
            enable_turn_detection=False,
        )
        for index, frame in enumerate(frames):
            playback_time = (index + 1) / frame_fps
            input_count += 1
            pending_frames += 1
            conversation.append_audio(base64.b64encode(silent_audio).decode("ascii"))
            conversation.append_video(base64.b64encode(frame.read_bytes()).decode("ascii"))
            time.sleep(frame_period)
            if playback_time >= next_trigger:
                create_response()
                next_trigger += QWEN_TRIGGER_INTERVAL_SECONDS
        if pending_frames:
            create_response()
    finally:
        conversation.close()
    return "".join(text_parts).strip(), stream_events, provider_events


def call_model(
    api_key: str | None,
    model_key: str,
    task: str,
    query: str,
    video_path: Path,
    frame_fps: int,
) -> tuple[str, str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="fitinteract_realtime_") as temp_dir:
        frames = extract_realtime_frames(video_path, Path(temp_dir), frame_fps)
        if model_key == "minicpm":
            response_text, stream_events, provider_events = minicpm_realtime_video(
                query, frames, frame_fps
            )
        else:
            if not api_key:
                raise RuntimeError("Qwen API key is required")
            response_text, stream_events, provider_events = qwen_realtime_video(
                api_key, query, frames, frame_fps
            )

    raw = json.dumps(
        {
            "response_text": response_text,
            "stream_events": stream_events,
            "provider_events": provider_events,
        },
        ensure_ascii=False,
    )
    if task == "reps":
        assembled = assemble_reps(stream_events)
    else:
        assembled = assemble_coaching(stream_events)
    details = {
        "model_key": model_key,
        "mode": (
            "minicpm_full_duplex_video"
            if model_key == "minicpm"
            else "qwen_manual_periodic_video"
        ),
        "frame_fps": frame_fps,
        "frame_width": REALTIME_FRAME_WIDTH,
        "frame_count": len(frames),
        "silent_audio_timeline": True,
        "response_trigger": (
            "model_listen_speak_decision"
            if model_key == "minicpm"
            else "periodic_client"
        ),
        "client_response_create": model_key == "qwen",
    }
    if model_key == "minicpm":
        details.update(
            endpoint=MODEL_SETTINGS["minicpm"]["realtime_url"],
            input_audio_format="float32_pcm_16000hz_mono",
            inference_chunk_s=MINICPM_INFERENCE_CHUNK_SECONDS,
            generate_audio=False,
            vad=False,
            force_listen_count=0,
            drain_pending_inference=True,
            **summarize_minicpm_output(provider_events),
        )
    else:
        details.update(trigger_interval_s=QWEN_TRIGGER_INTERVAL_SECONDS)
    return raw, assembled, details
