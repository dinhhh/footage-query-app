# Evaluation README

## 1. Overview

This README is for the **evaluation module only**.

The evaluation pipeline measures how well the video RAG system can answer event-specific questions using raw video input, then scores the generated answer against the human annotation ground truth.

The evaluation flow is:

```text
local video + annotation event
        ↓
generate or use user question
        ↓
run product inference through chat_with_raw_video_direct
        ↓
parse model answer into clean CSV fields
        ↓
judge output with local Ollama LLM-as-a-Judge
        ↓
export evaluation_outputs/video_eval_rows.csv
```

Important: the annotation file is used only as the **ground-truth baseline for evaluation**. It should not be used by the actual product inference pipeline to answer the question.

---

## 2. Folder Structure

The evaluation code is inside:

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

Main files:

```text
video_eval.py
```

Runs the full evaluation pipeline.

```text
llm_annotation.py
```

Loads annotation data, matches local videos, and creates evaluation cases.

```text
llm_judge.py
```

Runs local Ollama LLM-as-a-Judge.

```text
eval_csv.py
```

Exports an Excel-safe CSV file and prints summary metrics.

```text
eval_question_generator.py
```

Generates user questions from annotation events using a local Ollama model.

```text
eval_product_parser.py
```

Parses the product output into cleaner fields for the CSV.

```text
check_gemini_keys.py
```

Checks whether Gemini API keys are still usable.

---

## 3. Data and Annotation Placement

Put the raw video files and annotation file under:

```text
core_backend/Data/
```

Expected structure:

```text
core_backend/
└── Data/
    ├── Testing_Normal_Videos_Anomaly/
    │   ├── Shoplifting041_x264.mp4
    │   ├── Shoplifting042_x264.mp4
    │   └── ...
    └── UCFCrime_Test.json
```

The evaluation code recursively searches for video files inside `core_backend/Data/`, so videos can be inside subfolders such as:

```text
core_backend/Data/Testing_Normal_Videos_Anomaly/
```

The annotation file should be placed here:

```text
core_backend/Data/UCFCrime_Test.json
```

The video filename should match the annotation key. For example:

```text
Shoplifting041_x264.mp4
```

should match an annotation key like:

```text
Shoplifting041_x264
```

The annotation file is expected to include fields similar to:

```json
{
  "Shoplifting041_x264": {
    "duration": 120.0,
    "timestamps": [[10.0, 20.0]],
    "sentences": ["A person takes an item from the shop."]
  }
}
```

---

## 4. Environment Setup

Create and activate a virtual environment:

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

Make sure FFmpeg is installed because the evaluation uses `ffprobe` to get video duration:

```bash
ffprobe -version
```

On macOS, install with:

```bash
brew install ffmpeg
```

---

## 5. Setup Ollama

Check available local models:

```bash
ollama list
```

Example models used in this project:

```text
llama3:latest
gemma4:latest
```

If a model is missing, pull it:

```bash
ollama pull llama3
```

or:

```bash
ollama pull gemma4
```

The evaluation uses Ollama for:

```text
question generation
local LLM-as-a-Judge
```

Default Ollama URL:

```text
http://localhost:11434/api/chat
```

---

## 6. Setup API Key if the Product Pipeline Still Uses Gemini

The evaluation judge can run locally with Ollama.

However, if `llm_pipeline.py`, `video_pipeline.py`, or embedding code still calls Gemini, then you still need a Gemini API key.

You can store it in:

```text
.streamlit/secrets.toml
```

Example:

```toml
GOOGLE_API_KEY="YOUR_API_KEY"
```

or set it in the terminal:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

Some code may also look for:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

---

## 7. Check Whether Gemini Keys Still Work

If you want to check whether the Gemini API keys are still usable, run:

```bash
python3 -m core_backend.eval.check_gemini_keys
```

This is useful before running any pipeline component that still depends on Gemini.

If the output shows quota or authentication errors, update or rotate the key before running the API-dependent part.

---

## 8. Run Test on 1 Video

Run 1 video and 1 event first:

```bash
python3 -m core_backend.eval.video_eval -n 1 -e 1
```

Use a specific local judge model:

```bash
python3 -m core_backend.eval.video_eval -n 1 -e 1 -m gemma4:latest
```

or:

```bash
python3 -m core_backend.eval.video_eval -n 1 -e 1 -m llama3:latest
```

---

## 9. CLI Arguments

The main evaluation command is:

```bash
python3 -m core_backend.eval.video_eval
```

Useful arguments:

```text
-n, --limit-videos
```

Limit the number of videos to evaluate.

Example:

```bash
python3 -m core_backend.eval.video_eval -n 1
```

```text
-e, --limit-events
```

Limit the number of annotation events per video.

Example:

```bash
python3 -m core_backend.eval.video_eval -n 1 -e 1
```

```text
-q, --query
```

Use a custom query.

If omitted, local `llama3` generates questions from each annotation event.

```text
--questions-per-event
```

Number of local Llama-generated questions per annotation event.

```text
--append-output
```

Append rows to the existing CSV instead of overwriting it.

```text
-m, --judge-model
```

Local Ollama model used as LLM-as-a-Judge.

Example:

```bash
python3 -m core_backend.eval.video_eval -m gemma4:latest
```

```text
--question-generator-model
```

Local Ollama model used to generate user questions.

```text
--ollama-url
```

Ollama API endpoint. Default:

```text
http://localhost:11434/api/chat
```

```text
--sleep-sec
```

Sleep time between evaluation cases.

---

## 10. Run All Videos

Run all matched videos and all annotation events:

```bash
python3 -m core_backend.eval.video_eval
```

Run all using `gemma4:latest` as judge:

```bash
python3 -m core_backend.eval.video_eval -m gemma4:latest
```

Run all and append results to the existing CSV:

```bash
python3 -m core_backend.eval.video_eval --append-output
```

---

## 11. Output CSV

The evaluation output is saved to:

```text
evaluation_outputs/video_eval_rows.csv
```

The CSV is Excel-safe:

```text
- UTF-8 with BOM
- quoted fields
- no multiline cells
- model output split into separate columns
```

---

## 12. Evaluation Columns

Main columns:

```text
video_id
```

The video identifier matched from the local file and annotation key.

```text
video_duration_seconds
```

Total video duration extracted using `ffprobe`.

```text
event_index
```

The index of the annotation event inside the video.

```text
user_query
```

The question sent to the product pipeline.

```text
gold_start
gold_end
```

Ground-truth temporal range from the annotation file.

```text
annotation_answer
```

Human annotation event description.

```text
model_answer_summary
```

Cleaned final model answer used for judging.

```text
model_qa_question
model_qa_answer_summary
```

Question and answer parsed from the product chat history if available.

```text
model_occurrence_count
```

Number of occurrences returned by the product output.

```text
model_first_occurrence_start
model_first_occurrence_end
model_first_occurrence_description
```

First predicted event occurrence extracted from the model output if available.

```text
accuracy_relevance_score
```

Score from 0 to 5. Measures whether the model answer identifies the correct event.

```text
hallucination_score
```

Score from 0 to 5. Higher is better.

5 means no hallucination, 0 means mostly unsupported or invented.

```text
temporal_correctness_score
```

Score from 0 to 5. Measures whether the answer aligns with the correct time range.

```text
annotation_has_human
model_has_human
human_match
```

spaCy-based human/person detection and comparison.

```text
annotation_has_action
model_has_action
action_match
```

spaCy-based action/activity detection and comparison.

```text
event_found
```

Whether the judge believes the system found the target event.

```text
overall_pass
```

Final pass/fail judgement.

```text
latency_seconds
```

End-to-end product inference time.

```text
judge_explanation
```

Short explanation from the local LLM judge.

---

## 13. Notes

- The evaluation only uses annotations as ground truth after inference.
- Put both the raw video files and `UCFCrime_Test.json` under `core_backend/Data/`.
- Do not pass `gold_start` and `gold_end` into the product model as part of the answer generation process.
- If `Evaluation cases: 0`, check whether local video names match annotation keys.
- If Ollama fails, check that the model is installed with `ollama list`.
- If Gemini quota errors appear, the failing part is still using API somewhere in the product pipeline.
- If the CSV looks broken in Excel, use the latest `eval_csv.py`, which removes line breaks inside cells and quotes all fields.
