import re
from pathlib import Path


MODEL_SETTINGS = {
    "minicpm": {
        "model": "MiniCPM-o-4.5-Realtime",
        "base_url": "https://minicpmo45.modelbest.cn",
        "realtime_url": "wss://minicpmo45.modelbest.cn/v1/realtime?mode=video",
    },
    "qwen": {
        "model": "qwen3.5-omni-flash-realtime",
        "base_url": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
}

TASKS = ("reps", "bad_good", "accident", "multiplayer")
REALTIME_VIDEO_FPS = 10
REALTIME_FRAME_WIDTH = 540
MINICPM_INFERENCE_CHUNK_SECONDS = 1
QWEN_TRIGGER_INTERVAL_SECONDS = 0.5
RESPONSE_TIMEOUT_SECONDS = 60
QWEN_VOICE = "Ethan"
COUNT_WORDS = (
    "one two three four five six seven eight nine ten "
    "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
)
COUNT_RE = re.compile(rf"\b({'|'.join(COUNT_WORDS.split())}|\d+)\s*\.", re.I)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"I:\MVP demo")
SUMMARY_FIELDS = [
    "run_id", "result_path", "video_id", "task", "model", "api_success",
    "parse_success", "error", "gold_rep_count", "pred_rep_count", "count_exact",
    "count_absolute_error", "timing_correct_count", "timing_total", "predicted_event_count",
    "gold_actions", "pred_actions", "coach_timing_correct", "over_intervention",
]
