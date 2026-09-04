import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from realtime import minicpm_realtime_video, summarize_minicpm_output


class FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.incoming.put_nowait(json.dumps({"type": "session.queue_done"}))
        self.sent: list[dict] = []
        self.decision_events_delivered = 0

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        if message["type"] == "session.init":
            self.incoming.put_nowait(json.dumps({
                "type": "session.created", "session_id": "sess_test",
                "mode": "full_duplex",
            }))
        elif message["type"] == "input.append":
            self.incoming.put_nowait(json.dumps({
                "type": "response.output.delta", "kind": "listen",
                "input_id": message["input"]["input_id"],
                "metrics": {"n_tts_tokens": 0, "cost_tts_ms": 0},
            }))
        elif message["type"] == "session.close":
            if self.decision_events_delivered != 2:
                raise AssertionError("session closed before pending decisions were drained")
            self.incoming.put_nowait(json.dumps({
                "type": "session.closed", "reason": "user_stop",
            }))
            self.incoming.put_nowait(None)

    async def recv(self) -> str:
        raw = await self.incoming.get()
        if raw is None:
            raise StopAsyncIteration
        return raw

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            raw = await self.recv()
            if json.loads(raw).get("type") == "response.output.delta":
                self.decision_events_delivered += 1
            return raw
        except StopAsyncIteration:
            raise StopAsyncIteration from None


class FakeConnect:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *_args) -> None:
        return None


class MiniCPMRealtimeTest(unittest.TestCase):
    def test_official_video_protocol_uses_one_second_float32_chunks(self) -> None:
        websocket = FakeWebSocket()

        def connect(url: str, **kwargs):
            self.assertEqual(
                url,
                "wss://minicpmo45.modelbest.cn/v1/realtime?mode=video",
            )
            self.assertIsNone(kwargs["max_size"])
            return FakeConnect(websocket)

        async def no_sleep(_seconds: float) -> None:
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            frames = []
            for index in range(15):
                frame = Path(temp_dir) / f"frame_{index:04d}.jpg"
                frame.write_bytes(f"jpeg-{index}".encode())
                frames.append(frame)

            with (
                patch("websockets.connect", side_effect=connect),
                patch("realtime.asyncio.sleep", side_effect=no_sleep),
            ):
                _text, stream_events, _provider_events = minicpm_realtime_video(
                    "count reps", frames, frame_fps=10
                )

        init = next(message for message in websocket.sent if message["type"] == "session.init")
        self.assertEqual(init["payload"]["system_prompt"], "count reps")
        self.assertEqual(init["payload"]["use_tts"], False)
        self.assertEqual(init["payload"]["config"]["generate_audio"], False)
        self.assertEqual(init["payload"]["config"]["force_listen_count"], 0)

        appends = [message for message in websocket.sent if message["type"] == "input.append"]
        self.assertEqual([len(message["input"]["video_frames"]) for message in appends], [10, 5])
        self.assertEqual(
            [message["input"]["input_id"] for message in appends],
            ["decision_1", "decision_2"],
        )
        self.assertTrue(all(
            len(base64.b64decode(message["input"]["audio"])) == 16000 * 4
            for message in appends
        ))
        self.assertTrue(all(message["input"]["force_listen"] is False for message in appends))
        self.assertEqual(websocket.sent[-1]["type"], "session.close")
        self.assertNotIn("response.create", [message["type"] for message in websocket.sent])
        serialized = json.dumps(websocket.sent)
        self.assertNotIn("turn_detection", serialized)
        self.assertNotIn("server_vad", serialized)
        self.assertEqual(
            [event["decision"] for event in stream_events],
            ["decision #1", "decision #2"],
        )
        self.assertEqual([event["time_s"] for event in stream_events], [1, 2])
        audit = summarize_minicpm_output(_provider_events)
        self.assertEqual(audit["kind_audio_count"], 0)
        self.assertEqual(audit["audio_output_chunks"], 0)
        self.assertEqual(audit["n_tts_tokens"], 0)
        self.assertEqual(audit["cost_tts_ms"], 0)
        self.assertTrue(audit["tts_disabled_verified"])

    def test_tts_audit_rejects_real_backend_audio_work(self) -> None:
        audit = summarize_minicpm_output([
            {
                "type": "response.output.delta",
                "kind": "listen",
                "decision_index": 1,
                "metrics": {"n_tts_tokens": 0},
            },
            {
                "type": "response.output.delta",
                "kind": "text",
                "decision_index": 2,
                "metrics": {
                    "n_tts_tokens": 21,
                    "cost_tts_ms": 260.379,
                },
            },
            {
                "type": "response.output.delta",
                "kind": "audio",
                "decision_index": 2,
                "metrics": {
                    "n_tts_tokens": 21,
                    "cost_tts_ms": 260.379,
                },
            },
        ])

        self.assertEqual(audit["kind_audio_count"], 1)
        self.assertEqual(audit["n_tts_tokens"], 21)
        self.assertEqual(audit["cost_tts_ms"], 260.379)
        self.assertFalse(audit["tts_disabled_verified"])


if __name__ == "__main__":
    unittest.main()
