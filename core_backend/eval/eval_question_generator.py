# ============================================================
# IMPORTS
# ============================================================

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .eval_utils import call_ollama_chat, clean_question, compact_text, parse_json_object


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class GeneratedQuestion:
    question_index: int
    question_type: str
    question: str


# ============================================================
# QUESTION TYPE CONFIG
# ============================================================

QUESTION_TYPES = [
    "event_retrieval",
    "relevance_accuracy",
    "temporal_correctness",
]


# ============================================================
# VALIDATION HELPERS
# ============================================================

FORBIDDEN_GENERIC_PATTERNS = [
    r"\bsummary\b",
    r"\bsummarize\b",
    r"\boverall\b",
    r"\bwhole video\b",
    r"\bentire video\b",
    r"\bwhat happens in the video\b",
    r"\bwhat happened in the video\b",
    r"\bwhat is happening in the video\b",
    r"\bwhat's happening in the video\b",
    r"\bdescribe the video\b",
    r"\btell me about the video\b",
]

FORBIDDEN_TIME_WINDOW_PATTERNS = [
    r"\bfrom\s+\d+(\.\d+)?\s*(s|sec|secs|second|seconds)?\s+(to|until|-)\s+\d+",
    r"\bbetween\s+\d+(\.\d+)?\s*(s|sec|secs|second|seconds)?\s+and\s+\d+",
    r"\bin\s+\d+(\.\d+)?\s*(s|sec|secs|second|seconds)\s+(to|until|-)\s+\d+",
    r"\b\d+(\.\d+)?\s*(s|sec|secs|second|seconds)\s+(to|until|-)\s+\d+",
    r"\bwhat happened\s+(from|between|in)\s+\d+",
    r"\bwhat is happening\s+(from|between|in)\s+\d+",
    r"\bhow many\b.*\b(from|between|in)\s+\d+",
]

TEMPORAL_REQUIRED_PATTERNS = [
    r"\bwhen\b",
    r"\bwhat time\b",
    r"\btimestamp\b",
    r"\btime stamp\b",
    r"\btime range\b",
    r"\boccur\b",
    r"\bhappen\b",
]


def contains_forbidden_pattern(question: str) -> bool:
    """
    Detect bad generated questions, such as:
    - summary questions
    - whole-video questions
    - questions that reveal a time window
    """
    question_lower = question.lower()

    for pattern in FORBIDDEN_GENERIC_PATTERNS + FORBIDDEN_TIME_WINDOW_PATTERNS:
        if re.search(pattern, question_lower):
            return True

    return False


def extract_annotation_keywords(annotation_answer: str) -> List[str]:
    """
    Extract lightweight keywords from the annotation.

    This is used only to avoid overly generic questions.
    """
    text = compact_text(annotation_answer).lower()

    stopwords = {
        "the",
        "and",
        "that",
        "this",
        "there",
        "where",
        "when",
        "with",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "video",
        "shows",
        "showing",
        "seen",
        "visible",
        "appears",
        "appeared",
        "person",
        "people",
        "someone",
        "somebody",
        "event",
        "activity",
    }

    tokens = re.findall(r"[a-zA-Z]+", text)

    keywords = []
    for token in tokens:
        token = token.strip().lower()

        if len(token) < 4:
            continue

        if token in stopwords:
            continue

        keywords.append(token)

    # Keep unique order
    seen = set()
    unique_keywords = []

    for keyword in keywords:
        if keyword in seen:
            continue

        seen.add(keyword)
        unique_keywords.append(keyword)

    return unique_keywords[:8]


def question_mentions_event(question: str, annotation_answer: str) -> bool:
    """
    Check whether the question is likely about the annotated event.

    This is intentionally soft:
    - If annotation has useful keywords, require at least one keyword.
    - If annotation has no useful keywords, allow the question.
    """
    keywords = extract_annotation_keywords(annotation_answer)

    if not keywords:
        return True

    question_lower = question.lower()

    return any(keyword in question_lower for keyword in keywords)


def is_temporal_question(question: str) -> bool:
    """
    Check whether a temporal_correctness question actually asks for time.
    """
    question_lower = question.lower()

    return any(
        re.search(pattern, question_lower)
        for pattern in TEMPORAL_REQUIRED_PATTERNS
    )


def is_valid_question(
    question: str,
    question_type: str,
    annotation_answer: str,
) -> bool:
    """
    Validate generated question.

    The main goal is to prevent:
    - summary-style questions
    - leaked time-window questions
    - vague temporal questions
    """
    question = clean_question(question)

    if not question:
        return False

    if contains_forbidden_pattern(question):
        return False

    if question_type in ["event_retrieval", "relevance_accuracy"]:
        if not question_mentions_event(question, annotation_answer):
            return False

    if question_type == "temporal_correctness":
        if not is_temporal_question(question):
            return False

        if not question_mentions_event(question, annotation_answer):
            return False

    return True


# ============================================================
# FALLBACK QUESTIONS
# ============================================================

def build_fallback_question_by_type(
    annotation_answer: str,
    question_type: str,
) -> GeneratedQuestion:
    """
    Build one safe deterministic fallback question for a specific question type.
    """
    annotation_answer = compact_text(annotation_answer)

    if question_type == "event_retrieval":
        question = (
            "Can you find the specific moment in the video where this event occurs: "
            f"{annotation_answer}"
        )

    elif question_type == "relevance_accuracy":
        question = (
            "What exactly happens during this event in the video: "
            f"{annotation_answer}"
        )

    elif question_type == "temporal_correctness":
        question = (
            "At approximately what timestamp does this event occur in the video: "
            f"{annotation_answer}"
        )

    else:
        question = (
            "Can you answer a question about this specific event in the video: "
            f"{annotation_answer}"
        )

    return GeneratedQuestion(
        question_index=0,
        question_type=question_type,
        question=clean_question(question),
    )


def build_fallback_questions(
    annotation_answer: str,
    num_questions: int = 3,
) -> List[GeneratedQuestion]:
    """
    Deterministic fallback if local Llama fails.

    The fallback questions focus only on:
    1. event_retrieval
    2. relevance_accuracy
    3. temporal_correctness

    No summary-style questions.
    No timestamp leakage.
    """
    selected_types = QUESTION_TYPES[:num_questions]

    questions = [
        build_fallback_question_by_type(
            annotation_answer=annotation_answer,
            question_type=question_type,
        )
        for question_type in selected_types
    ]

    for idx, question in enumerate(questions, start=1):
        question.question_index = idx

    return questions


# ============================================================
# LOCAL OLLAMA QUESTION GENERATOR
# ============================================================

class OllamaQuestionGenerator:
    """
    Generate natural user questions from annotation using local Ollama.

    This is used only to create evaluation queries.
    It does not call Gemini API.
    """

    def __init__(
        self,
        model: str = "llama3:latest",
        ollama_url: str = "http://localhost:11434/api/chat",
        max_retries: int = 2,
        sleep_base: float = 2.0,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.sleep_base = sleep_base

    def build_prompt(
        self,
        annotation_answer: str,
        annotation_start: float,
        annotation_end: float,
        num_questions: int = 3,
    ) -> str:
        """
        Build prompt for local Llama question generation.

        The generated questions must test:
        1. event_retrieval
        2. relevance_accuracy
        3. temporal_correctness

        The model must not generate summary-style questions or reveal the ground-truth timestamp.
        """
        prompt = f"""
You are helping evaluate a CCTV video retrieval and video question-answering system.

Your task:
Generate exactly {num_questions} natural user questions from the human annotation.

The questions must focus only on these three evaluation aspects:

1. event_retrieval
- Test whether the system can retrieve/find the specific annotated event.
- The question must be about the event itself, not the whole video.
- Good style:
  "Can you find the moment where the vehicle accident happens?"
  "Is there a moment where a person steals an item?"
- Bad style:
  "What is happening in the video?"
  "Can you summarize the video?"

2. relevance_accuracy
- Test whether the system can correctly describe the annotated event.
- The question should focus on the key actors, objects, location, and action.
- The question should help reveal hallucinated, missing, or irrelevant details.
- Good style:
  "What happens during the accident involving the vehicle?"
  "What does the person do near the store shelf?"
- Bad style:
  "What happened overall?"
  "What is the summary?"

3. temporal_correctness
- Test whether the system can identify when the annotated event happened.
- The question must ask for the approximate timestamp or time range of the event.
- The question must focus on the specific event.
- Do NOT give any time window in the question.
- Do NOT ask what happened between two timestamps.
- Good style:
  "At approximately what timestamp does the vehicle accident occur?"
  "When does the person steal the item?"
- Bad style:
  "What happened from 6s to 10s?"
  "What is happening between 6 seconds and 10 seconds?"
  "How many cars are there from 6s to 10s?"

Human annotation:
{annotation_answer}

Ground-truth time range for your understanding only:
Start: {annotation_start:.2f} seconds
End: {annotation_end:.2f} seconds

Important:
The ground-truth time range is only for your understanding.
You must not reveal, mention, or hint at this time range in any generated question.

Strict rules:
- Do NOT generate any summary question.
- Do NOT ask about the whole video.
- Do NOT ask "What happened in the video?"
- Do NOT ask "What is happening in the video?"
- Do NOT ask "What happened from X seconds to Y seconds?"
- Do NOT ask "What is happening between X seconds and Y seconds?"
- Do NOT ask "How many objects were there from X seconds to Y seconds?"
- Do NOT mention any timestamp from the annotation.
- Do NOT reveal or hint at the ground-truth time range.
- Do NOT copy the annotation sentence word-for-word.
- Each question must focus on the annotated event.
- Keep each question concise and natural.
- Each question must be answerable from the video.
- Use only the annotation as the source of truth.
- Return only valid JSON.
- Do not include markdown.
- Do not include explanations outside JSON.

Return exactly this JSON schema:

{{
  "questions": [
    {{
      "question_type": "event_retrieval",
      "question": ""
    }},
    {{
      "question_type": "relevance_accuracy",
      "question": ""
    }},
    {{
      "question_type": "temporal_correctness",
      "question": ""
    }}
  ]
}}
"""
        return prompt.strip()

    def parse_questions(
        self,
        raw_text: str,
        annotation_answer: str,
        num_questions: int = 3,
    ) -> List[GeneratedQuestion]:
        """
        Parse local Llama output into GeneratedQuestion objects.

        This also validates each question. If Llama generates a bad question
        such as a summary question or a leaked time-window question, it is
        replaced by a safe fallback question.
        """
        parsed = parse_json_object(raw_text)
        questions_raw = parsed.get("questions", [])

        if not isinstance(questions_raw, list):
            return build_fallback_questions(annotation_answer, num_questions)

        selected_types = QUESTION_TYPES[:num_questions]
        questions_by_type: Dict[str, GeneratedQuestion] = {}

        for item in questions_raw:
            if not isinstance(item, dict):
                continue

            question_type = str(item.get("question_type", "")).strip().lower()
            question = clean_question(item.get("question", ""))

            if question_type not in selected_types:
                continue

            if question_type in questions_by_type:
                continue

            if not is_valid_question(
                question=question,
                question_type=question_type,
                annotation_answer=annotation_answer,
            ):
                continue

            questions_by_type[question_type] = GeneratedQuestion(
                question_index=0,
                question_type=question_type,
                question=question,
            )

        final_questions: List[GeneratedQuestion] = []

        for question_type in selected_types:
            question = questions_by_type.get(question_type)

            if question is None:
                question = build_fallback_question_by_type(
                    annotation_answer=annotation_answer,
                    question_type=question_type,
                )

            final_questions.append(question)

        for idx, question in enumerate(final_questions, start=1):
            question.question_index = idx

        return final_questions

    def generate_questions(
        self,
        annotation_answer: str,
        annotation_start: float,
        annotation_end: float,
        num_questions: int = 3,
    ) -> List[GeneratedQuestion]:
        """
        Generate evaluation questions using local Llama.
        Falls back to safe deterministic questions if Ollama fails.
        """
        prompt = self.build_prompt(
            annotation_answer=annotation_answer,
            annotation_start=annotation_start,
            annotation_end=annotation_end,
            num_questions=num_questions,
        )

        for attempt in range(self.max_retries):
            try:
                raw_text = call_ollama_chat(
                    prompt=prompt,
                    model=self.model,
                    ollama_url=self.ollama_url,
                    temperature=0.5,
                    timeout=180,
                )

                questions = self.parse_questions(
                    raw_text=raw_text,
                    annotation_answer=annotation_answer,
                    num_questions=num_questions,
                )

                if questions:
                    return questions

            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"[WARN] Local question generator failed: {e}")
                    return build_fallback_questions(annotation_answer, num_questions)

                time.sleep(self.sleep_base ** attempt)

        return build_fallback_questions(annotation_answer, num_questions)