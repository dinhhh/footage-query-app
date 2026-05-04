# ============================================================
# IMPORTS
# ============================================================

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .eval_utils import call_ollama_chat, clamp_score, parse_json_object, to_bool


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class JudgeResult:
    accuracy_relevance_score: int
    hallucination_score: int
    temporal_correctness_score: int
    event_found: bool
    overall_pass: bool
    explanation: str


# ============================================================
# JUDGE RESULT HELPERS
# ============================================================

def build_failed_result(message: str) -> JudgeResult:
    """
    Return a safe failed judge result.
    """
    return JudgeResult(
        accuracy_relevance_score=0,
        hallucination_score=5,
        temporal_correctness_score=0,
        event_found=False,
        overall_pass=False,
        explanation=message,
    )


def judge_result_to_dict(result: JudgeResult) -> Dict[str, Any]:
    """
    Convert JudgeResult dataclass into dictionary.
    """
    return asdict(result)


# ============================================================
# LOCAL OLLAMA JUDGE CLASS
# ============================================================

class LLMJudge:
    """
    Local LLM-as-a-Judge evaluator using Ollama.
    """

    def __init__(
        self,
        model: str = "gemma4:latest",
        ollama_url: str = "http://localhost:11434/api/chat",
        max_retries: int = 3,
        sleep_base: float = 2.0,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.sleep_base = sleep_base

    # ============================================================
    # JUDGE PROMPT
    # ============================================================

    def build_prompt(
        self,
        user_query: str,
        annotation_answer: str,
        annotation_start: float,
        annotation_end: float,
        system_output: str,
    ) -> str:
        """
        Build local judge prompt.

        This version is intentionally more tolerant:
        - Extra descriptive details are allowed if they do not contradict the annotation.
        - Hallucination is mainly penalized when the main person/actor or main action is clearly different.
        - Temporal correctness allows approximately +/- 5 seconds as fully correct.
        """
        prompt = f"""
You are an impartial but tolerant evaluator for a multimodal video understanding system.

The system receives a natural language query, retrieves or analyses a relevant video segment, and generates a final answer.

Your task is to evaluate the final system output against the human annotation.

Important context:
- The user query may be generated from the annotation event description.
- The user query does not include the annotation timestamp.
- The system must identify whether the event occurs and provide an approximate time if visible.
- The annotation timestamp below is only for evaluation.

Human Annotation:
{annotation_answer}

Annotation Temporal Range:
Start: {annotation_start:.2f} seconds
End: {annotation_end:.2f} seconds

User Query:
{user_query}

System Output:
{system_output}

General judging rules:
- Use the human annotation as the verified reference.
- Do not require exact wording.
- Judge semantic meaning, not exact phrasing.
- The system output may contain more details than the annotation. Extra details are allowed if they are plausible and do not clearly contradict the annotation.
- Do not penalize the system just because it provides a longer answer than the annotation.
- Do not over-penalize minor wording differences, minor object descriptions, or harmless background details.
- Penalize strongly only when the answer describes a clearly different main action, a clearly different main person/actor, or a clearly different event.
- If the annotation event is normal/no anomaly and the system says no dangerous/suspicious activity, this can be considered correct.
- Return only valid JSON.
- Do not include markdown.
- Do not include extra explanation outside JSON.

Evaluate using these criteria:

1. accuracy_relevance_score:
Score from 0 to 5.
Measures whether the system correctly identifies the main event/activity described in the annotation.

Use this guide:
- 5: The main event is correctly identified.
- 4: The event is mostly correct, with small missing or vague details.
- 3: The answer is partially related but incomplete or somewhat unclear.
- 2: The answer mentions something related but misses the main event.
- 1: The answer is barely related.
- 0: The answer describes a different event or does not answer the query.

2. hallucination_score:
Score from 0 to 5.
0 means no hallucination.
5 means the answer is mostly unsupported or invented.

Important relaxed hallucination rule:
- Do NOT mark hallucination just because the system gives more details than the annotation.
- Do NOT mark hallucination for plausible extra context, background objects, or harmless descriptive details.
- Only increase hallucination significantly if the system output clearly changes the main action, main actor/person, or main event.
- If the system correctly identifies the event but adds minor uncertain details, keep hallucination_score low.

Use this guide:
- 0: No meaningful hallucination. The event/action/person matches the annotation.
- 1: Very minor extra details, but no contradiction to the main event.
- 2: Some unsupported details, but the main person/action/event is still correct.
- 3: Noticeable unsupported details that may affect interpretation, but not a completely different event.
- 4: Major contradiction in person, object, or action.
- 5: Mostly invented, describes a different person/action/event, or contradicts the annotation completely.

3. temporal_correctness_score:
Score from 0 to 5.
Measures whether the system refers to the correct time range or relevant segment.

Temporal tolerance:
- If the model timestamp/range overlaps the annotation range, it should receive high temporal credit.
- If the model timestamp is within approximately 5 seconds before annotation_start or 5 seconds after annotation_end, treat it as correct.
- Exact timestamps are not required.
- Approximate phrases such as "near the beginning", "around the middle", or "later in the video" can receive partial credit if they are reasonably consistent with the annotation.
- If the system correctly identifies the event but gives no timestamp, temporal correctness should be partial, not full.
- If the system identifies the wrong event, temporal_correctness_score should be low even if it gives a timestamp.

Use this guide:
- 5: Timestamp/range overlaps the annotation range, or is within +/- 5 seconds of the annotation range.
- 4: Timing is close but slightly outside the +/- 5 second tolerance.
- 3: Timing is broadly correct but vague, such as a reasonable part of the video.
- 2: Event is correct but timing is missing, very vague, or noticeably off.
- 1: Timing is mostly wrong, but the answer is still somewhat related.
- 0: Wrong event, no usable timing, or timing is completely unrelated.

4. event_found:
true if the system correctly detects the main event or correctly identifies normal activity.
false if it misses the main event or describes a clearly different event.

5. overall_pass:
true if the answer is useful and mostly correct.
Use a tolerant standard:
- overall_pass can be true if the main event is correct, hallucination_score is 0, 1, or 2, and temporal correctness is reasonable.
- Do not fail the answer only because it has small extra details.
- Do not fail the answer only because the timestamp is slightly off within about +/- 5 seconds.
- overall_pass should be false if the main action or main person is clearly wrong, or hallucination_score is 4 or 5.

6. explanation:
A short explanation of the judgement. Mention whether the main event/action/person matched and whether the timing was acceptable.

Return only valid JSON using this exact schema:

{{
  "accuracy_relevance_score": 0,
  "hallucination_score": 0,
  "temporal_correctness_score": 0,
  "event_found": false,
  "overall_pass": false,
  "explanation": ""
}}
"""
        return prompt.strip()

    # ============================================================
    # RESULT PARSING
    # ============================================================

    def parse_result(self, raw_text: str) -> JudgeResult:
        """
        Convert raw local judge response into JudgeResult.
        """
        parsed = parse_json_object(raw_text)

        if not parsed:
            return build_failed_result(
                f"Local judge returned invalid JSON: {raw_text[:300]}"
            )

        return JudgeResult(
            accuracy_relevance_score=clamp_score(
                parsed.get("accuracy_relevance_score", 0)
            ),
            hallucination_score=clamp_score(
                parsed.get("hallucination_score", 5)
            ),
            temporal_correctness_score=clamp_score(
                parsed.get("temporal_correctness_score", 0)
            ),
            event_found=to_bool(parsed.get("event_found", False)),
            overall_pass=to_bool(parsed.get("overall_pass", False)),
            explanation=str(parsed.get("explanation", "")).strip(),
        )

    # ============================================================
    # JUDGE MODEL CALL
    # ============================================================

    def judge(
        self,
        user_query: str,
        annotation_answer: str,
        annotation_start: float,
        annotation_end: float,
        system_output: str,
    ) -> JudgeResult:
        """
        Run local Ollama judge and return structured result.
        """
        prompt = self.build_prompt(
            user_query=user_query,
            annotation_answer=annotation_answer,
            annotation_start=annotation_start,
            annotation_end=annotation_end,
            system_output=system_output,
        )

        for attempt in range(self.max_retries):
            try:
                raw_text = call_ollama_chat(
                    prompt=prompt,
                    model=self.model,
                    ollama_url=self.ollama_url,
                    temperature=0.0,
                    timeout=300,
                )
                return self.parse_result(raw_text)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return build_failed_result(f"Local judge failed: {e}")

                time.sleep(self.sleep_base ** attempt)

        return build_failed_result("Local judge failed after retries.")