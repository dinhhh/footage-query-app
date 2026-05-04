# ============================================================
# IMPORTS
# ============================================================

import json
from typing import Any, Dict, List

from .eval_utils import compact_text, parse_json_any, try_literal_eval


# ============================================================
# PRODUCT OUTPUT PARSING
# ============================================================

def extract_answer_from_json_obj(obj: Any) -> str:
    """
    Extract direct model answer from product JSON object.

    Expected product output example:
    {
        "event_found": true,
        "overall_summary": "...",
        "occurrences": [...]
    }
    """
    if not isinstance(obj, dict):
        return ""

    preferred_keys = [
        "overall_summary",
        "model_answer",
        "answer",
        "final_answer",
        "response",
        "result",
        "description",
        "explanation",
    ]

    for key in preferred_keys:
        value = obj.get(key)

        if isinstance(value, str) and value.strip():
            return compact_text(value)

    for _, value in obj.items():
        if isinstance(value, str) and value.strip():
            return compact_text(value)

    return ""


def normalise_occurrence_value(value: Any) -> List[Any]:
    """
    Convert occurrences into a list.
    """
    if isinstance(value, list):
        return value

    if value is None or value == "":
        return []

    return [value]


def extract_first_occurrence_time_fields(occurrences: List[Any]) -> Dict[str, str]:
    """
    Extract start/end time from the first model occurrence only.

    The detailed occurrence description is intentionally not saved to keep CSV concise.
    """
    result = {
        "model_first_occurrence_start": "",
        "model_first_occurrence_end": "",
    }

    if not occurrences:
        return result

    first = occurrences[0]

    if not isinstance(first, dict):
        return result

    start = (
        first.get("start")
        or first.get("start_time")
        or first.get("start_seconds")
        or first.get("start_timestamp")
        or first.get("timestamp_start")
        or ""
    )

    end = (
        first.get("end")
        or first.get("end_time")
        or first.get("end_seconds")
        or first.get("end_timestamp")
        or first.get("timestamp_end")
        or ""
    )

    result["model_first_occurrence_start"] = str(start)
    result["model_first_occurrence_end"] = str(end)

    return result


def parse_answer_json(answer_text: str) -> Dict[str, Any]:
    """
    Parse model answer JSON and extract useful CSV fields only.
    """
    parsed = parse_json_any(answer_text)

    result = {
        "model_answer": compact_text(answer_text),
        "model_occurrence_count": 0,
        "model_first_occurrence_start": "",
        "model_first_occurrence_end": "",
    }

    if isinstance(parsed, dict):
        model_answer = extract_answer_from_json_obj(parsed)
        occurrences = normalise_occurrence_value(parsed.get("occurrences"))

        result["model_answer"] = model_answer or compact_text(answer_text)
        result["model_occurrence_count"] = len(occurrences)
        result.update(extract_first_occurrence_time_fields(occurrences))

    return result


def parse_chat_history(history: Any) -> Dict[str, str]:
    """
    Parse chat history only as fallback.

    We do not save model_qa_question because it duplicates user_query.
    We do not save model_qa_answer_summary because it duplicates model_answer.
    """
    result = {
        "model_answer": "",
    }

    if not isinstance(history, list) or not history:
        return result

    last_item = history[-1]

    if not isinstance(last_item, dict):
        return result

    answer = last_item.get("answer", "")
    parsed_answer = parse_answer_json(answer)

    result["model_answer"] = parsed_answer.get("model_answer", "")

    return result


def parse_product_output(raw_output: Any) -> Dict[str, Any]:
    """
    Parse output from chat_with_raw_video_direct into non-duplicated fields.

    Handles:
    - Raw string
    - JSON string
    - Tuple: (answer_json_string, chat_history_list)
    - String representation of tuple
    - Dict
    - List chat history
    """
    result = {
        "model_answer": "",
        "model_occurrence_count": 0,
        "model_first_occurrence_start": "",
        "model_first_occurrence_end": "",
    }

    literal_value = raw_output

    if isinstance(raw_output, str):
        possible_literal = try_literal_eval(raw_output)

        if possible_literal is not None:
            literal_value = possible_literal

    if isinstance(literal_value, tuple):
        answer_part = literal_value[0] if len(literal_value) > 0 else ""
        history_part = literal_value[1] if len(literal_value) > 1 else []

        answer_result = parse_answer_json(str(answer_part))
        history_result = parse_chat_history(history_part)

        result.update(answer_result)

        if not result["model_answer"]:
            result["model_answer"] = history_result.get("model_answer", "")

        return result

    if isinstance(literal_value, list):
        history_result = parse_chat_history(literal_value)
        result.update(history_result)
        return result

    if isinstance(literal_value, dict):
        answer_result = parse_answer_json(
            json.dumps(literal_value, ensure_ascii=False)
        )
        result.update(answer_result)
        return result

    answer_result = parse_answer_json(str(literal_value))
    result.update(answer_result)

    return result