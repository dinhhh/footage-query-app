# ============================================================
# IMPORTS
# ============================================================

import ast
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================
# TEXT HELPERS
# ============================================================

def compact_text(text: Any) -> str:
    """
    Convert text into a clean one-line string.
    """
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)

    return text


def clean_question(text: Any) -> str:
    """
    Clean generated question and ensure it ends with '?'.
    """
    text = compact_text(text)

    if not text:
        return ""

    if not text.endswith("?"):
        text = text.rstrip(".") + "?"

    return text


# ============================================================
# JSON / LITERAL HELPERS
# ============================================================

def strip_markdown_json(text: Any) -> str:
    """
    Remove markdown JSON fences if present.
    """
    text = str(text or "").strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    return text


def parse_json_object(text: Any) -> Dict[str, Any]:
    """
    Parse JSON object from model output.

    Returns {} if parsing fails or if parsed JSON is not a dictionary.
    """
    if not text:
        return {}

    text = strip_markdown_json(text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

    return {}


def parse_json_any(text: Any) -> Optional[Any]:
    """
    Parse any valid JSON from model output.

    Returns None if parsing fails.
    """
    if not text:
        return None

    text = strip_markdown_json(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def try_literal_eval(value: Any) -> Optional[Any]:
    """
    Try to parse Python literal string such as:
    ('answer', [{'question': ..., 'answer': ...}])
    """
    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    if not (value.startswith("(") or value.startswith("[") or value.startswith("{")):
        return None

    try:
        return ast.literal_eval(value)
    except Exception:
        return None


# ============================================================
# VALUE HELPERS
# ============================================================

def to_bool(value: Any) -> bool:
    """
    Convert true/false-like value into bool.
    """
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in [
        "true",
        "1",
        "yes",
        "y",
        "pass",
        "correct",
    ]


def to_float(value: Any, default: float = 0.0) -> float:
    """
    Convert value into float safely.
    """
    try:
        if value is None or value == "":
            return default

        return float(value)

    except Exception:
        return default


def clamp_score(value: Any, minimum: int = 0, maximum: int = 5) -> int:
    """
    Convert score into integer between minimum and maximum.
    """
    try:
        value = int(value)
    except Exception:
        value = maximum

    return max(minimum, min(maximum, value))


# ============================================================
# VIDEO HELPERS
# ============================================================

def get_video_duration_seconds(video_path: Path) -> float:
    """
    Get video duration in seconds using ffprobe.

    Returns 0.0 if ffprobe fails.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        duration = result.stdout.strip()

        if not duration:
            return 0.0

        return round(float(duration), 3)

    except Exception:
        return 0.0


# ============================================================
# OLLAMA HELPER
# ============================================================

def call_ollama_chat(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float = 0.0,
    timeout: int = 300,
) -> str:
    """
    Call local Ollama chat API and return message content.
    """
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "options": {
            "temperature": temperature,
        },
    }

    req = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result.get("message", {}).get("content", "").strip()