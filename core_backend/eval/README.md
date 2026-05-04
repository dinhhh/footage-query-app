# Evaluation Module README

This README is for the **evaluation module only**. It explains how to run the evaluation pipeline for the Hybrid Multimodal RAG video system, not the full Streamlit application.

The evaluation pipeline checks whether the system can answer video-based questions correctly by comparing the final model output against the UCF-Crime annotation ground truth. It supports local Ollama models for question generation and LLM-as-a-Judge evaluation, while the main video-answering pipeline can still use Gemini if your `llm_pipeline.py` is configured that way.

---

## 1. Overview

The evaluation workflow is:

```text
UCF-Crime annotation
        ↓
Generate user questions from each annotation event
        ↓
Run chat_with_raw_video_direct() on the raw video
        ↓
Parse the model answer into clean CSV fields
        ↓
Use local Ollama LLM-as-a-Judge to score the answer
        ↓
Export evaluation_outputs/video_eval_rows.csv
```

The annotation file is used only as the **ground-truth baseline** for evaluation. It should not be used by the product pipeline during normal inference, except for generating evaluation questions.

---

## 2. Folder Structure

Current evaluation folder:

```text
core_backend/eval/
├── __init__.py
├── check_gemini_keys.py
├── eval_csv.py
├── eval_metrics.py
├── eval_product_parser.py
├── eval_question_generator.py
├── eval_utils.py
├── gemini_key_manager.py
├── llm_annotation.py
├── llm_judge.py
└── video_eval.py
```

Suggested responsibility of each file:

| File | Purpose |
|---|---|
| `video_eval.py` | Main entry point for running evaluation |
| `llm_annotation.py` | Loads UCF-Crime annotations and builds evaluation cases |
| `eval_question_generator.py` | Generates user questions from annotation events using local Ollama |
| `eval_product_parser.py` | Parses product output into clean fields for CSV |
| `llm_judge.py` | Runs local Ollama LLM-as-a-Judge scoring |
| `eval_csv.py` | Writes Excel-safe CSV output |
| `eval_metrics.py` | Computes evaluation summary metrics |
| `eval_utils.py` | Shared helper functions |
| `gemini_key_manager.py` | Manages Gemini keys if Gemini is still used in the product pipeline |
| `check_gemini_keys.py` | Utility for checking Gemini key availability |

---

## 3. Environment Setup

From the project root:

```bash
cd footage-query-app
```

Create and activate a virtual environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If spaCy is not installed yet:

```bash
pip install spacy
python3 -m spacy download en_core_web_sm
```

Make sure FFmpeg is installed because the evaluator uses `ffprobe` to read video duration:

```bash
ffmpeg -version
ffprobe -version
```

On macOS, install with:

```bash
brew install ffmpeg
```

---

## 4. Setup Ollama

The evaluation uses local Ollama models for question generation and LLM-as-a-Judge.

Check installed models:

```bash
ollama list
```

Example available models:

```text
llama3:latest
gemma4:latest
```

Pull a model if needed:

```bash
ollama pull llama3
```

Make sure Ollama is running:

```bash
ollama serve
```

If Ollama is already running, this command may show a port-in-use message. That is fine if `ollama list` works.

---

## 5. Setup API Key if the Pipeline Still Uses Gemini

The evaluation judge and question generator can run locally with Ollama, but your product pipeline may still call Gemini inside `llm_pipeline.py` or another backend file.

If your video-answering pipeline still uses Gemini, add your key to:

```text
.streamlit/secrets.toml
```

Example:

```toml
GOOGLE_API_KEY="YOUR_KEY_HERE"
```

Also make sure `.streamlit/secrets.toml` is included in `.gitignore`:

```gitignore
.streamlit/secrets.toml
```

If the product pipeline is fully local, this Gemini key setup may not be needed.

---

## 6. Run a 1-Video Test

Run one video and one event first:

```bash
python3 -m core_backend.eval.video_eval -n 1 -e 1
```

Using a specific local judge model:

```bash
python3 -m core_backend.eval.video_eval -n 1 -e 1 -m gemma4:latest
```

This is the recommended first test before running the full evaluation.

---

## 7. CLI Arguments

The main script is:

```bash
python3 -m core_backend.eval.video_eval
```

Available arguments:

| Argument | Meaning |
|---|---|
| `-n`, `--limit-videos` | Limit the number of videos to evaluate |
| `-e`, `--limit-events` | Limit the number of annotation events per video |
| `-q`, `--query` | Custom query. If omitted, local `llama3` generates questions from each annotation event |
| `--questions-per-event` | Number of locally generated questions per annotation event |
| `--append-output` | Append rows to the existing CSV instead of overwriting it |
| `-m`, `--judge-model` | Local Ollama model used as LLM-as-a-Judge |
| `--question-generator-model` | Local Ollama model used to generate user questions |
| `--ollama-url` | Ollama API endpoint. Default is `http://localhost:11434/api/chat` |
| `--sleep-sec` | Delay between evaluation cases |

Example with generated questions:

```bash
python3 -m core_backend.eval.video_eval \
  -n 1 \
  -e 1 \
  --questions-per-event 3 \
  -m gemma4:latest \
  --question-generator-model llama3:latest
```

Example with a custom query:

```bash
python3 -m core_backend.eval.video_eval \
  -n 1 \
  -e 1 \
  -q "Is there any suspicious activity in this video?" \
  -m gemma4:latest
```

Example appending new results:

```bash
python3 -m core_backend.eval.video_eval \
  -n 5 \
  -e 2 \
  --append-output
```

---

## 8. Run All Videos

Run the full evaluation:

```bash
python3 -m core_backend.eval.video_eval
```

Run all with a specific judge model:

```bash
python3 -m core_backend.eval.video_eval -m gemma4:latest
```

Run all with 3 generated questions per event:

```bash
python3 -m core_backend.eval.video_eval \
  --questions-per-event 3 \
  -m gemma4:latest \
  --question-generator-model llama3:latest
```

The full run may take a long time because each evaluation case can involve:

```text
video retrieval / answer generation
+ local question generation
+ local LLM judge scoring
```

---

## 9. Output CSV

The evaluation exports:

```text
evaluation_outputs/video_eval_rows.csv
```

The CSV is written in an Excel-safe format:

- UTF-8 with BOM
- quoted fields
- line breaks inside cells removed
- long JSON fields removed or simplified

---

## 10. Evaluation Column Meaning

| Column | Meaning |
|---|---|
| `video_id` | Video identifier matched with the annotation file |
| `video_duration_seconds` | Full video duration from `ffprobe` |
| `event_index` | Index of the annotation event in the video |
| `user_query` | Query sent to the video system |
| `question_type` | Type/category of generated question, if enabled |
| `gold_start` | Ground-truth event start time |
| `gold_end` | Ground-truth event end time |
| `annotation_answer` | Human annotation description |
| `model_answer_summary` | Cleaned final answer from the video system |
| `model_qa_question` | Question stored in the product chat history |
| `model_qa_answer_summary` | Cleaned answer stored in the product chat history |
| `model_occurrence_count` | Number of occurrences returned by the product |
| `model_first_occurrence_start` | First predicted occurrence start time |
| `model_first_occurrence_end` | First predicted occurrence end time |
| `model_first_occurrence_description` | Description of first predicted occurrence |
| `accuracy_relevance_score` | Judge score for semantic event correctness |
| `hallucination_score` | Judge score for unsupported/invented details |
| `temporal_correctness_score` | Judge score for time alignment |
| `annotation_has_human` | Whether the annotation mentions a human |
| `model_has_human` | Whether the model answer mentions a human |
| `human_match` | Whether annotation/model agree on human presence |
| `annotation_has_action` | Whether the annotation mentions an action |
| `model_has_action` | Whether the model answer mentions an action |
| `action_match` | Whether annotation/model agree on action presence |
| `event_found` | Whether the judge believes the event was correctly found |
| `overall_pass` | Final pass/fail judgement |
| `latency_seconds` | End-to-end product inference time |
| `judge_explanation` | Short explanation from the judge |

---

## 11. Notes

- This README describes only the evaluation pipeline in `core_backend/eval/`.
- The annotation file is used as ground truth for evaluation.
- If no custom query is provided, questions are generated from each annotation event.
- The generated question should not include the ground-truth timestamp.
- If `Evaluation cases: 0`, check whether local video names match the annotation keys.
- If Ollama fails, check `ollama list` and confirm the model name is correct.
- If the CSV looks broken in Excel, regenerate it with the latest `eval_csv.py`.
- Run `-n 1 -e 1` before running all videos.
- For a full run, expect longer processing time because each event/question needs product inference and judge scoring.

