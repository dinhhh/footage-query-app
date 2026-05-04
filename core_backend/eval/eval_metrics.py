# ============================================================
# IMPORTS
# ============================================================

from typing import Any, Dict, List

from .eval_utils import to_bool, to_float
from .llm_annotation import EvalCase, has_action, has_human


# ============================================================
# ROW-LEVEL METRICS
# ============================================================

def build_eval_row(
    case: EvalCase,
    question_index: int,
    question_type: str,
    user_query: str,
    video_duration_seconds: float,
    product_result: Dict[str, Any],
    judge_result,
) -> Dict[str, Any]:
    """
    Build one CSV row from one evaluation question.
    """
    model_answer = product_result.get("model_answer", "")

    model_has_human = has_human(model_answer)
    model_has_action = has_action(model_answer)

    human_match = bool(case.annotation_has_human == model_has_human)
    action_match = bool(case.annotation_has_action == model_has_action)

    return {
        # Basic video information
        "video_id": case.video_id,
        "video_duration_seconds": video_duration_seconds,
        "event_index": case.event_index,
        "question_index": question_index,
        "question_type": question_type,

        # Query and time comparison
        "user_query": user_query,
        "annotation_start": case.annotation_start,
        "annotation_end": case.annotation_end,
        "model_first_occurrence_start": product_result.get(
            "model_first_occurrence_start",
            "",
        ),
        "model_first_occurrence_end": product_result.get(
            "model_first_occurrence_end",
            "",
        ),

        # Annotation and model answer
        "annotation_answer": case.annotation_answer,
        "model_answer": model_answer,
        "model_occurrence_count": product_result.get("model_occurrence_count", 0),

        # Judge scores
        "accuracy_relevance_score": judge_result.accuracy_relevance_score,
        "hallucination_score": judge_result.hallucination_score,
        "temporal_correctness_score": judge_result.temporal_correctness_score,

        # Human/action comparison
        "annotation_has_human": str(case.annotation_has_human).lower(),
        "model_has_human": str(model_has_human).lower(),
        "human_match": str(human_match).lower(),
        "annotation_has_action": str(case.annotation_has_action).lower(),
        "model_has_action": str(model_has_action).lower(),
        "action_match": str(action_match).lower(),

        # Judge decision
        "judge_event_found": str(judge_result.event_found).lower(),
        "judge_overall_pass": str(judge_result.overall_pass).lower(),

        # Runtime and explanation
        "latency_seconds": product_result.get("latency_seconds", ""),
        "judge_explanation": judge_result.explanation,
    }


# ============================================================
# DATASET-LEVEL SUMMARY METRICS
# ============================================================

def build_eval_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build evaluation summary from all rows.
    """
    total = len(rows)

    judge_event_found_count = sum(
        1 for row in rows
        if to_bool(row.get("judge_event_found"))
    )

    judge_overall_pass_count = sum(
        1 for row in rows
        if to_bool(row.get("judge_overall_pass"))
    )

    human_match_count = sum(
        1 for row in rows
        if to_bool(row.get("human_match"))
    )

    action_match_count = sum(
        1 for row in rows
        if to_bool(row.get("action_match"))
    )

    total_latency_seconds = sum(
        to_float(row.get("latency_seconds"))
        for row in rows
        if str(row.get("latency_seconds", "")).strip() != ""
    )

    return {
        "rows_evaluated": total,
        "judge_event_found_count": judge_event_found_count,
        "judge_overall_pass_count": judge_overall_pass_count,
        "human_match_count": human_match_count,
        "action_match_count": action_match_count,
        "judge_event_found_rate": judge_event_found_count / total if total else 0.0,
        "judge_overall_pass_rate": judge_overall_pass_count / total if total else 0.0,
        "human_match_rate": human_match_count / total if total else 0.0,
        "action_match_rate": action_match_count / total if total else 0.0,
        "average_latency_seconds": total_latency_seconds / total if total else 0.0,
    }