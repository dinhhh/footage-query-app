# ============================================================
# IMPORTS
# ============================================================

import csv
import re
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# CSV CONFIG
# ============================================================

CSV_FIELDNAMES = [
    # Basic video information
    "video_id",
    "video_duration_seconds",
    "event_index",
    "question_index",
    "question_type",

    # Query and time comparison
    "user_query",
    "annotation_start",
    "annotation_end",
    "model_first_occurrence_start",
    "model_first_occurrence_end",

    # Annotation and model answer
    "annotation_answer",
    "model_answer",
    "model_occurrence_count",

    # Judge scores
    "accuracy_relevance_score",
    "hallucination_score",
    "temporal_correctness_score",

    # Human/action comparison
    "annotation_has_human",
    "model_has_human",
    "human_match",
    "annotation_has_action",
    "model_has_action",
    "action_match",

    # Judge decision
    "judge_event_found",
    "judge_overall_pass",

    # Runtime and explanation
    "latency_seconds",
    "judge_explanation",
]


# ============================================================
# CSV CLEANING HELPERS
# ============================================================

def clean_csv_value(value: Any) -> str:
    """
    Make values safer for Excel CSV import.
    """
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and align one row to CSV_FIELDNAMES.
    """
    return {
        field: clean_csv_value(row.get(field, ""))
        for field in CSV_FIELDNAMES
    }


# ============================================================
# CSV WRITING
# ============================================================

def save_eval_csv(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    filename: str = "video_eval_rows.csv",
    append: bool = False,
) -> Path:
    """
    Save evaluation rows to an Excel-friendly CSV.

    append=False:
        overwrite the CSV file.

    append=True:
        append new rows to the existing CSV file.
        If the file does not exist yet, write the header first.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / filename
    cleaned_rows = [clean_csv_row(row) for row in rows]

    file_exists = csv_path.exists()
    write_header = not append or not file_exists

    mode = "a" if append else "w"

    with open(csv_path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDNAMES,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )

        if write_header:
            writer.writeheader()

        writer.writerows(cleaned_rows)

    return csv_path


# ============================================================
# SUMMARY PRINTING
# ============================================================

def print_eval_summary(csv_path: Path, summary: Dict[str, Any]) -> None:
    """
    Print evaluation summary for the current run.
    """
    print("\n=== DONE ===")
    print(f"CSV saved to: {csv_path}")
    print(f"Rows evaluated in this run: {summary['rows_evaluated']}")
    print(f"Judge event found rate: {summary['judge_event_found_rate']:.4f}")
    print(f"Judge overall pass rate: {summary['judge_overall_pass_rate']:.4f}")
    print(f"Human match rate: {summary['human_match_rate']:.4f}")
    print(f"Action match rate: {summary['action_match_rate']:.4f}")
    print(f"Average latency seconds: {summary['average_latency_seconds']:.4f}")