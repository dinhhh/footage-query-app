# ============================================================
# IMPORTS
# ============================================================

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import spacy


# ============================================================
# CONFIG
# ============================================================

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ============================================================
# SPACY MODEL LOADING
# ============================================================

def load_spacy_model():
    """
    Load spaCy English model.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise OSError(
            "spaCy model en_core_web_sm is not installed.\n"
            "Run this command:\n"
            "python3 -m spacy download en_core_web_sm"
        ) from e


NLP = load_spacy_model()


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class EvalCase:
    video_id: str
    video_path: str
    event_index: int
    annotation_start: float
    annotation_end: float
    annotation_answer: str
    annotation_has_human: bool
    annotation_has_action: bool


# ============================================================
# VIDEO ID NORMALISATION
# ============================================================

def normalise_video_id(value: str) -> str:
    """
    Convert video filename/path into a stable lowercase video id.

    Example:
    Robbery001_x264.mp4 -> robbery001_x264
    """
    if value is None:
        return ""

    value = str(value).strip()
    value = Path(value).name

    for ext in VIDEO_EXTS:
        if value.lower().endswith(ext):
            value = value[: -len(ext)]
            break

    return value.strip().lower()


def display_video_id(value: str) -> str:
    """
    Keep readable video id for CSV output.
    """
    if value is None:
        return ""

    value = str(value).strip()
    value = Path(value).name

    for ext in VIDEO_EXTS:
        if value.lower().endswith(ext):
            value = value[: -len(ext)]
            break

    return value.strip()


# ============================================================
# ANNOTATION LOADING
# ============================================================

def load_annotations(path: Path) -> Any:
    """
    Load annotation JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOCAL VIDEO INDEX
# ============================================================

def build_local_video_map(data_dir: Path) -> Dict[str, Path]:
    """
    Build map:
    normalised_video_id -> actual local video path.
    """
    video_map: Dict[str, Path] = {}

    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in VIDEO_EXTS:
            continue

        video_map[normalise_video_id(p.name)] = p

    return video_map


# ============================================================
# ANNOTATION FORMAT NORMALISATION
# ============================================================

def get_video_id_from_item(item: Dict[str, Any]) -> str:
    """
    Extract video id from one annotation item if annotation file is list-based.
    """
    possible_keys = [
        "video_id",
        "video",
        "video_name",
        "filename",
        "file_name",
        "name",
    ]

    for key in possible_keys:
        if key in item and item[key]:
            return display_video_id(str(item[key]))

    return ""


def annotation_to_video_items(annotation_data: Any) -> List[Tuple[str, Any]]:
    """
    Convert different annotation JSON formats into:
    [(video_id, payload), ...]
    """
    items: List[Tuple[str, Any]] = []

    if isinstance(annotation_data, dict):
        for video_id, payload in annotation_data.items():
            items.append((display_video_id(video_id), payload))

    elif isinstance(annotation_data, list):
        for item in annotation_data:
            if not isinstance(item, dict):
                continue

            video_id = get_video_id_from_item(item)

            if video_id:
                items.append((video_id, item))

    else:
        raise ValueError("Unsupported annotation JSON format. Expected dict or list.")

    return items


# ============================================================
# VIDEO MATCHING
# ============================================================

def match_annotations(
    annotations: Any,
    data_dir: Path,
) -> List[Tuple[str, Path, Any]]:
    """
    Match annotation video ids with local video files.
    """
    video_map = build_local_video_map(data_dir)
    annotation_items = annotation_to_video_items(annotations)

    matched: List[Tuple[str, Path, Any]] = []

    for video_id, payload in annotation_items:
        key = normalise_video_id(video_id)

        if key in video_map:
            matched.append((display_video_id(video_id), video_map[key], payload))

    print(f"Local videos found: {len(video_map)}")
    print(f"Annotation videos found: {len(annotation_items)}")
    print(f"Matched videos: {len(matched)}")

    if len(matched) == 0:
        print("\n[DEBUG] First 10 local video ids:")
        for key in list(video_map.keys())[:10]:
            print(" -", key)

        print("\n[DEBUG] First 10 annotation video ids:")
        for video_id, _ in annotation_items[:10]:
            print(" -", normalise_video_id(video_id))

    return matched


# ============================================================
# EVENT EXTRACTION HELPERS
# ============================================================

def clean_annotation_sentence(text: str) -> str:
    """
    Clean annotation sentence for judge prompt.
    """
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)

    return text


def read_float(value: Any, default: float = 0.0) -> float:
    """
    Safely read float.
    """
    try:
        if value is None or value == "":
            return default

        return float(value)

    except Exception:
        return default


def get_text_from_event(event: Dict[str, Any]) -> str:
    """
    Extract event description from different possible keys.
    """
    possible_keys = [
        "sentence",
        "sentences",
        "description",
        "annotation",
        "event_description",
        "label",
        "event",
        "action",
        "class",
        "category",
    ]

    for key in possible_keys:
        value = event.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


# ============================================================
# EVENT EXTRACTION
# ============================================================

def iter_events(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Read annotated events from one video payload.

    Supports:
    - timestamps + sentences
    - events list
    - start/end fields
    """
    if not isinstance(payload, dict):
        return

    timestamps = payload.get("timestamps", None)
    sentences = payload.get("sentences", None)

    if isinstance(timestamps, list) and isinstance(sentences, list):
        for idx, (ts, sent) in enumerate(zip(timestamps, sentences)):
            if not isinstance(ts, list) or len(ts) != 2:
                continue

            annotation_answer = clean_annotation_sentence(sent)

            if not annotation_answer:
                continue

            annotation_start = read_float(ts[0])
            annotation_end = read_float(ts[1])

            if annotation_end < annotation_start:
                annotation_start, annotation_end = annotation_end, annotation_start

            yield {
                "event_index": idx,
                "annotation_start": annotation_start,
                "annotation_end": annotation_end,
                "annotation_answer": annotation_answer,
            }

        return

    events = payload.get("events", None)

    if isinstance(events, list):
        for idx, event in enumerate(events):
            if not isinstance(event, dict):
                continue

            annotation_answer = clean_annotation_sentence(get_text_from_event(event))

            if not annotation_answer:
                annotation_answer = "Annotated event"

            annotation_start = read_float(
                event.get("start_time")
                or event.get("start")
                or event.get("start_seconds")
                or event.get("anomaly_start")
            )

            annotation_end = read_float(
                event.get("end_time")
                or event.get("end")
                or event.get("end_seconds")
                or event.get("anomaly_end")
            )

            if annotation_end < annotation_start:
                annotation_start, annotation_end = annotation_end, annotation_start

            yield {
                "event_index": idx,
                "annotation_start": annotation_start,
                "annotation_end": annotation_end,
                "annotation_answer": annotation_answer,
            }

        return

    annotation_answer = clean_annotation_sentence(get_text_from_event(payload))

    annotation_start = read_float(
        payload.get("start_time")
        or payload.get("start")
        or payload.get("start_seconds")
        or payload.get("anomaly_start")
    )

    annotation_end = read_float(
        payload.get("end_time")
        or payload.get("end")
        or payload.get("end_seconds")
        or payload.get("anomaly_end")
    )

    if annotation_answer:
        if annotation_end < annotation_start:
            annotation_start, annotation_end = annotation_end, annotation_start

        yield {
            "event_index": 0,
            "annotation_start": annotation_start,
            "annotation_end": annotation_end,
            "annotation_answer": annotation_answer,
        }


# ============================================================
# HUMAN AND ACTION DETECTION
# ============================================================

def has_human(text: str) -> bool:
    """
    Detect whether text mentions a human/person using spaCy.
    """
    text = str(text or "").strip()

    if not text:
        return False

    doc = NLP(text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return True

    human_terms = {
        "person",
        "people",
        "man",
        "woman",
        "boy",
        "girl",
        "human",
        "individual",
        "pedestrian",
        "suspect",
        "victim",
        "driver",
        "passenger",
        "customer",
        "worker",
        "guard",
        "thief",
        "shoplifter",
        "robber",
        "intruder",
        "someone",
        "somebody",
        "personnel",
        "officer",
        "employee",
        "staff",
    }

    human_pronouns = {
        "he",
        "she",
        "they",
        "him",
        "her",
        "them",
        "his",
        "hers",
        "their",
        "theirs",
    }

    for token in doc:
        lemma = token.lemma_.lower()
        lower = token.text.lower()

        if lemma in human_terms or lower in human_terms:
            return True

        if lower in human_pronouns:
            return True

    return False


def has_action(text: str) -> bool:
    """
    Detect whether text mentions an action/activity using spaCy.
    """
    text = str(text or "").strip()

    if not text:
        return False

    doc = NLP(text)

    action_nouns = {
        "activity",
        "action",
        "movement",
        "fight",
        "attack",
        "assault",
        "stealing",
        "shoplifting",
        "robbery",
        "burglary",
        "shooting",
        "explosion",
        "arson",
        "abuse",
        "arrest",
        "accident",
        "vandalism",
        "chasing",
        "escape",
        "fall",
        "running",
        "walking",
        "violence",
        "crime",
        "incident",
        "event",
        "theft",
        "intrusion",
    }

    ignored_verbs = {
        "be",
        "have",
        "do",
        "seem",
        "appear",
        "look",
    }

    for token in doc:
        lemma = token.lemma_.lower()
        lower = token.text.lower()

        if token.pos_ == "VERB" and lemma not in ignored_verbs:
            return True

        if lemma in action_nouns or lower in action_nouns:
            return True

    return False


# ============================================================
# BUILD EVALUATION CASES
# ============================================================

def build_eval_cases(
    data_dir: Path,
    annotation_path: Path,
    limit_videos: Optional[int] = None,
    limit_events: Optional[int] = None,
) -> List[EvalCase]:
    """
    Build end-to-end evaluation cases.
    """
    annotations = load_annotations(annotation_path)
    video_items = match_annotations(annotations, data_dir)

    if limit_videos is not None:
        video_items = video_items[:limit_videos]

    cases: List[EvalCase] = []

    for video_id, video_path, payload in video_items:
        events = list(iter_events(payload))

        if limit_events is not None:
            events = events[:limit_events]

        if not events:
            print(f"[DEBUG] No events extracted for video: {video_id}")

        for event in events:
            annotation_answer = clean_annotation_sentence(event["annotation_answer"])

            cases.append(
                EvalCase(
                    video_id=video_id,
                    video_path=str(video_path),
                    event_index=int(event["event_index"]),
                    annotation_start=float(event["annotation_start"]),
                    annotation_end=float(event["annotation_end"]),
                    annotation_answer=annotation_answer,
                    annotation_has_human=has_human(annotation_answer),
                    annotation_has_action=has_action(annotation_answer),
                )
            )

    print(f"Evaluation cases built: {len(cases)}")

    return cases


# ============================================================
# CONVERSION HELPERS
# ============================================================

def eval_case_to_dict(case: EvalCase) -> Dict[str, Any]:
    """
    Convert EvalCase dataclass into dictionary.
    """
    return asdict(case)