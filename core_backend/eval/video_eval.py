# ============================================================
# IMPORTS
# ============================================================

import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# WARNING FILTERS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*urllib3 v2 only supports OpenSSL.*",
)


import chromadb

from .eval_csv import print_eval_summary, save_eval_csv
from .eval_metrics import build_eval_row, build_eval_summary
from .eval_product_parser import parse_product_output
from .eval_question_generator import GeneratedQuestion, OllamaQuestionGenerator
from .eval_utils import compact_text, get_video_duration_seconds
from .gemini_key_manager import GeminiKeyManager, is_quota_or_rate_error
from .llm_annotation import EvalCase, build_eval_cases
from .llm_judge import LLMJudge


# ============================================================
# DEFAULT CONFIG
# ============================================================

DATA_DIR = Path("core_backend/Data")
ANNOTATION_PATH = Path("core_backend/Data/UCFCrime_Test.json")
OUTPUT_DIR = Path("evaluation_outputs")

DB_PATH = "./cctv_chroma_db"
COLLECTION_NAME = "direct_video_vectors"

DEFAULT_QUERY = "AUTO_GENERATED_QUESTIONS"

DEFAULT_JUDGE_MODEL = "gemma4:latest"
DEFAULT_QUESTION_GENERATOR_MODEL = "llama3:latest"

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_CHUNK_DURATION = 15.0
DEFAULT_SLEEP_SEC = 0.5
DEFAULT_QUESTIONS_PER_EVENT = 3


# ============================================================
# VIDEO AND LLM PIPELINE IMPORT
# ============================================================

def import_pipeline_modules():
    """
    Import existing project pipeline modules.

    Important:
    - Do not edit video_pipeline.py or llm_pipeline.py.
    - We patch their module-level `client` later using GeminiKeyManager.
    """
    errors = []

    try:
        video_module = importlib.import_module("core_backend.video_pipeline")
        llm_module = importlib.import_module("core_backend.llm_pipeline")

        return video_module, llm_module

    except Exception as e:
        errors.append(f"package import failed: {repr(e)}")

    try:
        video_module = importlib.import_module(
            "..video_pipeline",
            package=__package__,
        )
        llm_module = importlib.import_module(
            "..llm_pipeline",
            package=__package__,
        )

        return video_module, llm_module

    except Exception as e:
        errors.append(f"relative import failed: {repr(e)}")

    try:
        current_file = Path(__file__).resolve()
        core_backend_dir = current_file.parents[1]

        if str(core_backend_dir) not in sys.path:
            sys.path.insert(0, str(core_backend_dir))

        video_module = importlib.import_module("video_pipeline")
        llm_module = importlib.import_module("llm_pipeline")

        return video_module, llm_module

    except Exception as e:
        errors.append(f"direct import failed: {repr(e)}")

    raise ImportError(
        "Cannot import pipeline modules.\n\n"
        "Check these points:\n"
        "1. core_backend/video_pipeline.py exists and contains:\n"
        "   - ingest_raw_video_direct\n"
        "2. core_backend/llm_pipeline.py exists and contains:\n"
        "   - chat_with_raw_video_direct\n"
        "3. Run from project root using:\n"
        "   python3 -m core_backend.eval.video_eval\n\n"
        "Import errors:\n"
        + "\n".join(errors)
    )


# ============================================================
# CHROMADB HELPERS
# ============================================================

def make_collection():
    """
    Connect to ChromaDB collection using default config.
    """
    return chromadb.PersistentClient(path=DB_PATH).get_or_create_collection(
        name=COLLECTION_NAME
    )


def is_video_indexed(collection, video_id: str) -> bool:
    """
    Check if a video is already indexed in ChromaDB.
    """
    try:
        result = collection.get(where={"video_id": video_id}, limit=1)
        return bool(result.get("ids"))

    except Exception:
        return False


# ============================================================
# QUESTION GENERATION
# ============================================================

def build_custom_question(cli_query: str) -> List[GeneratedQuestion]:
    """
    Use user-provided CLI query as a single evaluation question.
    """
    return [
        GeneratedQuestion(
            question_index=1,
            question_type="custom",
            question=compact_text(cli_query),
        )
    ]


def resolve_questions_for_case(
    case: EvalCase,
    cli_query: str,
    question_generator: OllamaQuestionGenerator,
    questions_per_event: int,
) -> List[GeneratedQuestion]:
    """
    Use local Llama to generate natural evaluation questions.

    If -q is provided, use that custom query instead.
    """
    if cli_query != DEFAULT_QUERY:
        return build_custom_question(cli_query)

    return question_generator.generate_questions(
        annotation_answer=case.annotation_answer,
        annotation_start=case.annotation_start,
        annotation_end=case.annotation_end,
        num_questions=questions_per_event,
    )


# ============================================================
# PRODUCT INFERENCE
# ============================================================

def product_output_has_quota_error(product_result: Dict[str, object]) -> bool:
    """
    Detect quota/rate-limit error text from parsed product output.
    """
    return is_quota_or_rate_error(product_result.get("model_answer", ""))


def run_product_once(
    video_path: Path,
    user_query: str,
    chat_with_raw_video_direct,
    key_manager: Optional[GeminiKeyManager] = None,
) -> Dict[str, object]:
    """
    Run the Hybrid Multimodal RAG product once.
    """
    start_time = time.perf_counter()

    max_attempts = key_manager.key_count if key_manager else 1
    parsed_output: Dict[str, object] = {}

    for _ in range(max_attempts):
        try:
            raw_output = chat_with_raw_video_direct(user_query, str(video_path))
            parsed_output = parse_product_output(raw_output)

            if (
                key_manager is not None
                and product_output_has_quota_error(parsed_output)
                and key_manager.switch_to_next_key()
            ):
                print(
                    "[Gemini key rotation] Product output contains quota/rate "
                    "error text. Retrying product inference with next key..."
                )
                continue

            break

        except Exception as e:
            if (
                key_manager is not None
                and is_quota_or_rate_error(e)
                and key_manager.switch_to_next_key()
            ):
                print(
                    "[Gemini key rotation] Product inference raised quota/rate "
                    "error. Retrying with next key..."
                )
                continue

            parsed_output = {
                "model_answer": f"ERROR: Product inference failed: {e}",
                "model_occurrence_count": 0,
                "model_first_occurrence_start": "",
                "model_first_occurrence_end": "",
            }
            break

    if not parsed_output:
        parsed_output = {
            "model_answer": (
                "ERROR: Product inference failed because all Gemini API keys "
                "were exhausted."
            ),
            "model_occurrence_count": 0,
            "model_first_occurrence_start": "",
            "model_first_occurrence_end": "",
        }

    latency_seconds = time.perf_counter() - start_time
    parsed_output["latency_seconds"] = round(latency_seconds, 6)

    return parsed_output


# ============================================================
# ARGUMENT PARSER
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end video RAG evaluation using generated user questions, "
            "annotation ground truth, and local Ollama LLM-as-a-Judge."
        )
    )

    parser.add_argument("-n", "--limit-videos", type=int, default=None)
    parser.add_argument("-e", "--limit-events", type=int, default=None)

    parser.add_argument(
        "-q",
        "--query",
        default=DEFAULT_QUERY,
        help=(
            "Custom query. If omitted, local llama3 generates questions "
            "from each annotation event."
        ),
    )

    parser.add_argument(
        "--questions-per-event",
        type=int,
        default=DEFAULT_QUESTIONS_PER_EVENT,
        help="Number of local Llama-generated questions per annotation event.",
    )

    parser.add_argument(
    "--append-output",
    action="store_true",
    help="Append rows to existing CSV instead of overwriting it.",
)
    parser.add_argument(
        "-m",
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Local Ollama model used as LLM judge.",
    )

    parser.add_argument(
        "--question-generator-model",
        default=DEFAULT_QUESTION_GENERATOR_MODEL,
        help="Local Ollama model used to generate user questions.",
    )

    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
    )

    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=DEFAULT_SLEEP_SEC,
    )

    return parser


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:
    args = build_arg_parser().parse_args()

    data_dir = DATA_DIR.resolve()
    annotation_path = ANNOTATION_PATH.resolve()
    output_dir = OUTPUT_DIR.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    print("=== End-to-End Video Evaluation ===")
    print(f"Data folder: {data_dir}")
    print(f"Annotation file: {annotation_path}")
    print(f"Output folder: {output_dir}")
    print(f"Local judge model: {args.judge_model}")
    print(f"Local question generator model: {args.question_generator_model}")
    print(f"Questions per event: {args.questions_per_event}")

    key_manager = GeminiKeyManager.from_sources()
    key_manager.activate_current_key()

    print(f"Gemini API keys loaded: {key_manager.key_count}")
    print(f"Starting with Gemini {key_manager.current_key_label()}")

    cases = build_eval_cases(
        data_dir=data_dir,
        annotation_path=annotation_path,
        limit_videos=args.limit_videos,
        limit_events=args.limit_events,
    )

    print(f"Annotation events: {len(cases)}")

    if not cases:
        print("No evaluation cases found.")
        return

    video_module, llm_module = import_pipeline_modules()

    patched_modules = key_manager.patch_modules(
        video_module,
        llm_module,
    )

    if patched_modules:
        print("Rotating Gemini client patched into:")
        for module_name in patched_modules:
            print(f" - {module_name}")
    else:
        print("[WARN] No pipeline module client was patched.")

    ingest_raw_video_direct = video_module.ingest_raw_video_direct
    chat_with_raw_video_direct = llm_module.chat_with_raw_video_direct

    collection = make_collection()

    judge = LLMJudge(
        model=args.judge_model,
        ollama_url=args.ollama_url,
    )

    question_generator = OllamaQuestionGenerator(
        model=args.question_generator_model,
        ollama_url=args.ollama_url,
    )

    rows: List[Dict[str, object]] = []
    product_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
    duration_cache: Dict[str, float] = {}

    for case_idx, case in enumerate(cases, start=1):
        print(
            f"\n[Event {case_idx}/{len(cases)}] "
            f"video={case.video_id}, event_index={case.event_index}"
        )

        video_path = Path(case.video_path)

        if case.video_id in duration_cache:
            video_duration_seconds = duration_cache[case.video_id]
        else:
            video_duration_seconds = get_video_duration_seconds(video_path)
            duration_cache[case.video_id] = video_duration_seconds

        try:
            already_indexed = is_video_indexed(collection, case.video_id)

            if already_indexed:
                print("Video already indexed. Skip ingest.")
            else:
                print("Video not indexed. Ingesting video...")
                ingest_raw_video_direct(
                    str(video_path),
                    chunk_duration=DEFAULT_CHUNK_DURATION,
                )

                if not is_video_indexed(collection, case.video_id):
                    print(
                        "[WARN] Video still does not appear indexed after ingestion. "
                        "Check ingestion logs above for quota/errors or video_id mismatch."
                    )

        except Exception as e:
            print(f"[ERROR] Ingest check/ingest failed: {e}")

        generated_questions = resolve_questions_for_case(
            case=case,
            cli_query=args.query,
            question_generator=question_generator,
            questions_per_event=args.questions_per_event,
        )

        for generated_question in generated_questions:
            user_query = generated_question.question

            print(
                f"\n  [Question {generated_question.question_index}/"
                f"{len(generated_questions)}] "
                f"type={generated_question.question_type}"
            )
            print(f"  User query: {user_query}")

            cache_key = (case.video_id, user_query)

            if cache_key in product_cache:
                print("  Using cached product output for this video/query.")
                product_result = product_cache[cache_key]

            else:
                print("  Running product inference...")
                product_result = run_product_once(
                    video_path=video_path,
                    user_query=user_query,
                    chat_with_raw_video_direct=chat_with_raw_video_direct,
                    key_manager=key_manager,
                )
                product_cache[cache_key] = product_result

            model_answer = str(product_result.get("model_answer", ""))

            print("  Judging output with local Ollama...")

            judge_result = judge.judge(
                user_query=user_query,
                annotation_answer=case.annotation_answer,
                annotation_start=case.annotation_start,
                annotation_end=case.annotation_end,
                system_output=model_answer,
            )

            row = build_eval_row(
                case=case,
                question_index=generated_question.question_index,
                question_type=generated_question.question_type,
                user_query=user_query,
                video_duration_seconds=video_duration_seconds,
                product_result=product_result,
                judge_result=judge_result,
            )

            rows.append(row)

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    csv_path = save_eval_csv(
        rows=rows,
        output_dir=output_dir,
        append=args.append_output,
    )

    summary = build_eval_summary(rows)

    print_eval_summary(
        csv_path=csv_path,
        summary=summary,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()