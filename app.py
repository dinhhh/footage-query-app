"""
Streamlit chat UI (ChatGPT-style) with random replies, optional Google Drive video
upload, and clickable time ranges to seek the in-session video.
Run: streamlit run app.py
"""

from __future__ import annotations

import io
import mimetypes
import random
import re
import subprocess
import tempfile
import uuid
from typing import Any
import json

import streamlit as st

from drive_upload import upload_video_to_google_drive
from core_backend.llm_pipeline import chat_with_raw_video_direct

# --- Response generators ---
NOT_IN_VIDEO_RESPONSE = "This content does not exist in your video"


def _ensure_message_id(msg: dict[str, Any]) -> str:
    mid = msg.get("id")
    if mid is None:
        mid = str(uuid.uuid4())
        msg["id"] = mid
    return mid


def _play_on_click(**kwargs: Any) -> None:
    a = int(kwargs["a"])
    b = int(kwargs["b"])
    cid = st.session_state.active_chat_id
    # Use a counter so clicking the same segment twice still triggers a rerun
    count = st.session_state.get(f"seek_count_{cid}", 0) + 1
    st.session_state[f"seek_count_{cid}"] = count
    st.session_state[f"seek_range_{cid}"] = (a, b, count)


def render_play_button(
    a: int,
    b: int,
    msg_id: str,
    *,
    label: str | None = None,
) -> None:
    """Text styled as an inline link; uses the same seek callback as before."""
    cid = st.session_state.active_chat_id
    display = label if label is not None else f"{a}:{b}"
    st.button(
        display,
        key=f"tsseek_{cid}_{msg_id}",
        on_click=_play_on_click,
        kwargs={"a": a, "b": b},
        type="tertiary",
        help="Seek video to this range",
        width="content",
        use_container_width=False,
    )


# Matches MM:SS, HH:MM:SS, optional ranges, optional **bold** wrappers.
# Range separator: ASCII hyphen, en/em dash; optional spaces (e.g. 0:01-0:02 or 00:00:03 - 00:00:06).
_RANGE_SEP = r"\s*[-\u2013\u2014]\s*"
TIMESTAMP_PATTERN = re.compile(
    rf"(?:\*\*)?(\d{{1,2}}:\d{{2}}(?::\d{{2}})?)(?:{_RANGE_SEP}(\d{{1,2}}:\d{{2}}(?::\d{{2}})?))?(?:\*\*)?"
)
# Whole-text scan (no ** anchors); matches times inside ``**…**`` the same as test.extract_timestamps.
_TIMESTAMP_GLOBAL_PATTERN = re.compile(
    rf"(\d{{1,2}}:\d{{2}}(?::\d{{2}})?)(?:{_RANGE_SEP}(\d{{1,2}}:\d{{2}}(?::\d{{2}})?))?"
)


def _ts_part_to_seconds(part: str) -> int:
    nums = [int(p) for p in part.split(":")]
    if len(nums) == 3:
        h, m, s = nums
        return h * 3600 + m * 60 + s
    if len(nums) == 2:
        m, s = nums
        return m * 60 + s
    raise ValueError(f"Unexpected timestamp format: {part}")


def _match_to_seconds_range(m: re.Match) -> tuple[int, int]:
    a = _ts_part_to_seconds(m.group(1))
    if m.group(2):
        b = _ts_part_to_seconds(m.group(2))
        return a, b
    return a, a


def extract_timestamps(text: str) -> list[tuple[int, int]]:
    """
    All timestamp ranges in document order, as (start_sec, end_sec).
    Single times use the same value for start and end.
    """
    out: list[tuple[int, int]] = []
    for m in _TIMESTAMP_GLOBAL_PATTERN.finditer(text):
        out.append(_match_to_seconds_range(m))
    return out


def _segment_line(line: str) -> list[tuple[Any, ...]]:
    """Split one line into ('text', str) and ('seek', label, a, b) pieces."""
    segs: list[tuple[Any, ...]] = []
    last = 0
    for m in TIMESTAMP_PATTERN.finditer(line):
        if m.start() > last:
            segs.append(("text", line[last : m.start()]))
        a, b = _match_to_seconds_range(m)
        label = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
        segs.append(("seek", label, a, b))
        last = m.end()
    if last < len(line):
        segs.append(("text", line[last:]))
    return segs


_LEADING_JUNK = re.compile(r"^[\s\u200b\u200c\u200d\ufeff]+")


def _strip_leading_invisible(s: str) -> str:
    """Remove spaces / ZWSP / BOM so fragments like '\\u200b- **' classify as list markup."""
    return _LEADING_JUNK.sub("", s or "")


def _is_leading_list_marker_only_fragment(s: str) -> bool:
    """True when a split text chunk is only list markup (e.g. '-', '- ', '- **'), not real words."""
    t = _strip_leading_invisible(s)
    if not t:
        return True
    if re.search(r"[A-Za-z0-9]", t):
        return False
    if any(c in "()[]{}" for c in t):
        return False
    # ASCII hyphen, unicode dashes, bullets, leading * for list/emphasis
    if not re.match(r"^[-–—•*‧·]", t):
        return False
    return len(t) <= 24


def _drop_leading_list_marker_column(segs: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Remove first column when it is only '-'/bullet/bold opener so the row starts at the timestamp."""
    if len(segs) < 2 or segs[0][0] != "text":
        return segs
    if _is_leading_list_marker_only_fragment(segs[0][1]):
        return segs[1:]
    return segs


_WS_LINE = re.compile(r"^(\s*)")


def _neutralize_first_list_marker_line(text: str) -> str:
    """Hide the first markdown list bullet (-/*) so the opening line isn't a list item."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        m = _WS_LINE.match(line)
        if not m:
            continue
        indent = m.group(1)
        rest = line[len(indent) :]
        if rest.startswith("- ") or rest.startswith("* "):
            lines[i] = indent + "\u200b" + rest
            break
    return "\n".join(lines)


def _markdown_chat_fragment(
    s: str,
    start_time: int | None,
    end_time: int | None,
    chat: dict[str, Any] | None,
    *,
    download_key: str | None = None,
) -> None:
    """Render markdown first, then an optional link-styled clip download (same ffmpeg cut)."""
    m = _WS_LINE.match(s)
    if s.startswith(": "):
        s = s[2:]
    if not m:
        st.markdown(s)
        _clip_download_link_after_markdown(
            start_time, end_time, chat, download_key
        )
        return
    indent = m.group(1)
    rest = s[len(indent) :]
    if rest.startswith("- ") or rest.startswith("* "):
        rest = "\u200b" + rest
    st.markdown(indent + rest)
    _clip_download_link_after_markdown(
        start_time, end_time, chat, download_key
    )


def _clip_download_link_after_markdown(
    start_time: int | None,
    end_time: int | None,
    chat: dict[str, Any] | None,
    download_key: str | None,
) -> None:
    if (
        download_key is None
        or chat is None
        or start_time is None
        or end_time is None
        or not chat.get("video_bytes")
    ):
        return
    mime = chat.get("video_mime") or "video/mp4"
    clip_bytes = build_video_clip_bytes(
        chat["video_bytes"], int(start_time), int(end_time), mime
    )
    cid = st.session_state.active_chat_id
    if clip_bytes is not None:
        st.download_button(
            label="Download this video segment",
            data=clip_bytes,
            file_name=f"clip_{start_time}_{end_time}.mp4",
            mime=mime,
            key=f"dl_md_{cid}_{download_key}",
            type="tertiary",
            width="content",
            use_container_width=False,
            help=f"Download segment {start_time}s–{end_time}s",
        )
    else:
        st.caption("Clip unavailable (ffmpeg)")


def render_reply_with_seek_links(reply: str, msg_id: str, chat) -> None:
    """Show assistant reply parsed from structured JSON, rendering play buttons and descriptions."""
    
    # 1. Clean the string (in case the LLM wrapped the JSON in markdown code blocks)
    clean_reply = reply.strip()
    if clean_reply.startswith("```json"):
        clean_reply = clean_reply[7:-3].strip()
    elif clean_reply.startswith("```"):
        clean_reply = clean_reply[3:-3].strip()

    # 2. Parse the JSON
    try:
        data = json.loads(clean_reply)
    except json.JSONDecodeError:
        # Fallback: If it's not valid JSON for some reason, just render as raw markdown
        st.markdown(reply)
        return

    # 3. Render the Overall Summary first
    if "overall_summary" in data and data["overall_summary"]:
        st.markdown(f"**Summary:** {data['overall_summary']}")
        st.markdown("---")

    # 4. Handle cases where the event wasn't found
    if not data.get("event_found", False):
        st.info("No specific events matching your query were found in the footage.")
        return

    # Helper function to convert "MM:SS" or "HH:MM:SS" into integer seconds
    def to_seconds(t_str: str) -> int:
        parts = t_str.split(':')
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return int(t_str)
        except ValueError:
            return 0 # Fallback for malformed time strings

    # 5. Render the occurrences in a perfect two-column layout
    occurrences = data.get("occurrences", [])
    for ci, occ in enumerate(occurrences):
        start_str = occ.get("start_timestamp", "00:00")
        end_str = occ.get("end_timestamp", "00:00")
        desc = occ.get("detailed_description", "")
        
        start_sec = to_seconds(start_str)
        end_sec = to_seconds(end_str)
        
        # Lock in the two columns: Button on the left, Description on the right
        cols = st.columns([15, 85], gap="small", vertical_alignment="top")
        
        with cols[0]:
            label = f"▶ {start_str}"
            render_play_button(
                start_sec,
                end_sec,
                f"{msg_id}_btn_{ci}",
                label=label
            )
            
        with cols[1]:
            # Use your existing text fragment renderer
            _markdown_chat_fragment(
                desc, 
                start_sec, 
                end_sec, 
                chat, 
                download_key=f"{msg_id}_desc_{ci}"
            )


def render_reply_with_seek_links_old(reply: str, msg_id: str, chat) -> None:
    """Show assistant reply; plain timestamps become play/seek controls (like render_play_button)."""
    reply = _neutralize_first_list_marker_line(reply)
    if not TIMESTAMP_PATTERN.search(reply):
        st.markdown(reply)
        return

    for li, line in enumerate(reply.split("\n")):
        if line == "":
            st.markdown("")
            continue
        segs = _drop_leading_list_marker_column(_segment_line(line))
        if len(segs) == 1 and segs[0][0] == "text":
            _markdown_chat_fragment(
                segs[0][1], None, None, chat, download_key=None,
            )
            continue
        weights: list[int] = [15, 35]
        cols = st.columns(weights, gap="small", vertical_alignment="top")
        last_a: int | None = None
        last_b: int | None = None
        for ci, seg in enumerate(segs):
            with cols[ci]:
                if seg[0] == "text":
                    dk = f"{msg_id}_L{li}_T{ci}" if last_a is not None else None
                    _markdown_chat_fragment(
                        seg[1], last_a, last_b, chat, download_key=dk,
                    )
                else:
                    _, label, a, b = seg
                    last_a, last_b = int(a), int(b)
                    render_play_button(
                        int(a),
                        int(b),
                        f"{msg_id}_L{li}_C{ci}",
                        label=label,
                    )


@st.cache_data(show_spinner=False)
def build_video_clip_bytes(
    video_bytes: bytes, start_sec: int, end_sec: int, mime_type: str
) -> bytes | None:
    """Cut [start_sec, end_sec] clip with ffmpeg and return bytes."""
    if end_sec <= start_sec:
        end_sec = start_sec + 1

    ext = mimetypes.guess_extension(mime_type or "video/mp4") or ".mp4"

    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = f"{tmp_dir}/input{ext}"
        out_path = f"{tmp_dir}/clip{ext}"

        with open(in_path, "wb") as f:
            f.write(video_bytes)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", in_path,
            "-c", "copy",
            out_path,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(out_path, "rb") as f:
                return f.read()
        except Exception:
            return None


def generate_random_text_response(user_message: str) -> str:
    templates = [
        'Interesting point about: "{msg}". Here is a random thought: the weather is computational.',
        'You said "{msg}". I\'ll reply with something arbitrary: seven platypuses in a trench coat.',
        "Acknowledged: {msg}. Random line: static on the radio, but friendly.",
        'On "{msg}": a random sentence — the elevator music was written in binary.',
        "Re: {msg} — imagine a cloud made of toast. That's the vibe.",
        NOT_IN_VIDEO_RESPONSE,
    ]
    selected = random.choice(templates)
    if selected == NOT_IN_VIDEO_RESPONSE:
        return selected
    return selected.format(msg=user_message.strip() or "(empty message)")


def generate_random_timeframes(
    max_duration_sec: int = 30,
    window_max: int = 5,
    max_clips: int = 5,
) -> list[tuple[int, int]]:
    """Generate 1-5 random time ranges."""
    upper = max(10, max_duration_sec)
    num_clips = random.randint(1, min(max_clips, 5))
    ranges: list[tuple[int, int]] = []

    for _ in range(num_clips):
        start = random.randint(0, max(0, upper - 2))
        end = min(upper, start + random.randint(1, window_max))
        if end <= start:
            end = min(upper, start + 1)
        ranges.append((start, end))

    return sorted(ranges, key=lambda x: x[0])[:max_clips]


def _drive_folder_id() -> str | None:
    try:
        gd = st.secrets.get("google_drive")
        if isinstance(gd, dict):
            fid = gd.get("folder_id")
            if fid:
                return str(fid)
        return st.secrets.get("google_drive.folder_id", None)
    except Exception:
        return None


def _init_session() -> None:
    if "chat_order" not in st.session_state:
        st.session_state.chat_order = []
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat_id" not in st.session_state:
        cid = str(uuid.uuid4())
        st.session_state.chats[cid] = _new_chat_data()
        st.session_state.chat_order.append(cid)
        st.session_state.active_chat_id = cid


def _new_chat_data() -> dict[str, Any]:
    return {
        "title": "New chat",
        "messages": [],
        "video_bytes": None,
        "video_name": None,
        "video_mime": "video/mp4",
        "video_seek_sec": 0,
        "video_end_sec": None,
        "video_seek_generation": 0,
        "session_history": [],
        "assistant_pending": False,
        "drive_upload_pending": False,
    }


def _invoke_assistant_model(chat: dict[str, Any]) -> tuple[str, list[Any]]:
    """
    Long-running assistant call. Wrapped by st.spinner in the UI so slow runs show loading.
    Set ILAB_DEV_LLM_DELAY_SEC=5 (optional) to simulate latency without changing this file.
    """
    import os
    import time as _time

    delay = os.environ.get("ILAB_DEV_LLM_DELAY_SEC", "").strip()
    if delay:
        _time.sleep(min(float(delay), 120.0))

    prompt = chat.get("_turn_prompt", "")
    fname = chat.get("_turn_fname")
    session_history = chat.get("session_history", [])
    return chat_with_raw_video_direct(prompt, fname, session_history)
# for testing purpose
#     import time
#     time.sleep(5)
#     return """In this video, there is **1 white car** turning.
#     **Clip 1:**
# - **0:01-0:02**: A white car turns right.
# - **0:02-0:03**: A grey SUV turns right.
# - **0:04-0:05**: A white truck with green crates turns left.""", session_history


def _append_assistant_turn_result(
    chat: dict[str, Any], reply: str, session_history: list[Any],
) -> None:
    chat["session_history"] = session_history
    chat["assistant_pending"] = False
    chat.pop("_turn_prompt", None)
    chat.pop("_turn_fname", None)
    query_timestamps = extract_timestamps(reply)
    if reply == NOT_IN_VIDEO_RESPONSE:
        chat["messages"].append(
            {"role": "assistant", "content": reply, "id": str(uuid.uuid4())}
        )
    else:
        chat["messages"].append(
            {
                "role": "assistant",
                "content": reply,
                "timeframes": query_timestamps,
                "id": str(uuid.uuid4()),
            }
        )


def _active_chat() -> dict[str, Any]:
    return st.session_state.chats[st.session_state.active_chat_id]


def _new_chat() -> None:
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = _new_chat_data()
    st.session_state.chat_order.append(cid)
    st.session_state.active_chat_id = cid


def _delete_chat(cid: str) -> None:
    if cid not in st.session_state.chats:
        return
    st.session_state.chat_order = [x for x in st.session_state.chat_order if x != cid]
    del st.session_state.chats[cid]
    if not st.session_state.chat_order:
        _new_chat()
    elif st.session_state.active_chat_id == cid:
        st.session_state.active_chat_id = st.session_state.chat_order[-1]


def _apply_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top:1rem !important; padding-bottom:0 !important; max-width:none !important; }
            /* Sidebar buttons */
            .stSidebar [data-testid="stButton"] > button { border:none !important; background:transparent !important; box-shadow:none !important; border-radius:12px !important; color:inherit !important; text-align:left !important; justify-content:flex-start !important; padding-left:0.5rem !important; transition:background-color 0.2s ease; }
            .stSidebar [data-testid="stButton"] > button > div { justify-content:flex-start !important; text-align:left !important; width:100%; }
            .stSidebar [data-testid="stButton"] > button:hover { background:rgba(255,255,255,0.06) !important; }
            .stSidebar [data-testid="stButton"] > button:focus, .stSidebar [data-testid="stButton"] > button:active { outline:none !important; box-shadow:none !important; border:none !important; background:rgba(255,255,255,0.08) !important; }
            .stSidebar [data-testid="stExpander"] { border:none !important; box-shadow:none !important; background:transparent !important; }
            /* White seekbar */
            video::-webkit-media-controls-timeline { filter:brightness(10) invert(1) !important; }
            video::-webkit-media-controls-volume-slider { filter:brightness(10) invert(1) !important; }
            video::-webkit-media-controls-panel { background:rgba(0,0,0,0.55) !important; }
            /* Suppress flicker on rerun — hide skeleton loaders */
            [data-testid="stSkeleton"] { display: none !important; }
            /* Smooth fade-in for new content */
            [data-testid="stHorizontalBlock"] > div:last-child > * {
                animation: fadein 0.15s ease-in;
            }
            @keyframes fadein { from { opacity: 0.4; } to { opacity: 1; } }
            /* RIGHT chat column scrolls independently */
            [data-testid="stHorizontalBlock"] > div:last-child {
                max-height: calc(100vh - 140px) !important;
                overflow-y: auto !important;
                padding-bottom: 10px !important;
            }
            /* LEFT video column: match chat height, no scroll (centering via st.container stretch) */
            [data-testid="stHorizontalBlock"] > div:first-child {
                max-height: calc(100vh - 140px) !important;
                overflow: hidden !important;
            }

            /* Inline timestamp “links” (tertiary seek buttons); key prefix tsseek_ */
            [class*="st-key-tsseek_"] button {
                color: #60a5fa !important;
                text-decoration: underline !important;
                text-underline-offset: 0.12em !important;
                font-weight: 400 !important;
                padding: 0 0.25rem !important;
                min-height: unset !important;
                height: auto !important;
                line-height: 1.35 !important;
                white-space: nowrap !important;
            }
            [class*="st-key-tsseek_"] button:hover {
                color: #93c5fd !important;
                background: transparent !important;
            }
            [class*="st-key-tsseek_"] button:focus-visible {
                outline: 2px solid #60a5fa !important;
                outline-offset: 2px !important;
            }
            /* Clip download after markdown; key prefix dl_md_ */
            [class*="st-key-dl_md_"] button {
                color: #34d399 !important;
                text-decoration: underline !important;
                font-weight: 400 !important;
                padding: 0 !important;
                min-height: unset !important;
                justify-content: flex-start !important;
            }
            [class*="st-key-dl_md_"] button:hover {
                color: #6ee7b7 !important;
                background: transparent !important;
            }
            /* Timestamp + description rows: top-align; no per-column scroll / max-height */
            [data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stHorizontalBlock"] {
                align-items: flex-start !important;
            }
            [data-testid="stHorizontalBlock"] > div:last-child [data-testid="column"] {
                overflow: visible !important;
                max-height: none !important;
            }
            [data-testid="stHorizontalBlock"] > div:last-child [data-testid="column"] [data-testid="stMarkdownContainer"] {
                overflow: visible !important;
                max-height: none !important;
            }
            [data-testid="stHorizontalBlock"] > div:last-child [data-testid="column"] p {
                margin: 0 0 0.35rem 0 !important;
                line-height: 1.45 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_random_timeframe_seconds(
    max_duration_sec: int = 10, window_max: int = 5
) -> tuple[int, int]:
    """Pick a random inclusive second range [a; b] with a <= b."""
    upper = max(5, max_duration_sec)
    start = random.randint(0, max(0, upper - 2))
    end = min(upper, start + random.randint(1, window_max))
    if end <= start:
        end = min(upper, start + 1)
    return start, end


def main() -> None:
    st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")
    _apply_css()
    _init_session()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        if st.button("➕ New chat", use_container_width=True):
            _new_chat()
            st.rerun()
        st.markdown("**Your chats**")
        for cid in reversed(st.session_state.chat_order):
            ch = st.session_state.chats[cid]
            if not ch["messages"]:
                continue
            label = ch["title"][:36] + ("…" if len(ch["title"]) > 36 else "")
            c1, c2 = st.columns([6, 1], gap="small")
            with c1:
                if st.button(label, key=f"tab_{cid}", use_container_width=True):
                    st.session_state.active_chat_id = cid
                    st.rerun()
            with c2:
                if st.button("🧹", key=f"del_{cid}", use_container_width=True):
                    _delete_chat(cid)
                    st.rerun()

    chat = _active_chat()

    # ── Seek state ────────────────────────────────────────────────────────────
    import time as _time
    range_key = f"seek_range_{st.session_state.active_chat_id}"
    autoplay_js = ""
    if range_key in st.session_state:
        val = st.session_state.pop(range_key)
        a, b = int(val[0]), int(val[1])
        if b <= a:
            b = a + 1
        chat["video_seek_sec"] = a
        chat["video_end_sec"] = b
        token = int(_time.time() * 1000)
        autoplay_js = (
            "<script>/* token:" + str(token) + " */"
            "(function waitForVideo(){"
            "var vids=window.parent.document.querySelectorAll('video');"
            "var v=vids[vids.length-1];"
            "if(!v){setTimeout(waitForVideo,100);return;}"
            "var start=" + str(a) + ",end=" + str(b) + ";"
            "function go(){"
            "v.ontimeupdate=null;"
            "v.currentTime=start;"
            "v.play().catch(function(){});"
            "if(end>start){"
            "v.ontimeupdate=function(){"
            "if(v.currentTime>=end){v.pause();v.ontimeupdate=null;}"
            "};"
            "}"
            "}"
            "if(v.readyState>=1){go();}else{v.addEventListener('loadedmetadata',go,{once:true});}"
            "})();</script>"
        )

    SEEKBAR_CSS = (
        "<script>(function(){"
        "function styleVideo(){"
        "var vids=window.parent.document.querySelectorAll('video');"
        "if(!vids.length){setTimeout(styleVideo,100);return;}"
        "if(!document.getElementById('vid-seekbar-style')){"
        "var s=document.createElement('style');"
        "s.id='vid-seekbar-style';"
        "s.textContent="
        "'video::-webkit-media-controls-timeline{filter:brightness(10) invert(1)!important;} "
        "video::-webkit-media-controls-volume-slider{filter:brightness(10) invert(1)!important;} "
        "video::-webkit-media-controls-panel{background:rgba(0,0,0,0.55)!important;}';"
        "window.parent.document.head.appendChild(s);"
        "}}"
        "styleVideo();"
        "})();</script>"
    )

    # ── chat_input MUST be outside columns to stay docked at bottom ──────────
    chat_input_value = st.chat_input(
        "Message...",
        accept_file=True,
        file_type=["mp4", "webm", "mov", "mkv", "avi"],
    )

    # Phase 1: append user message + context (defer Drive upload + LLM to later runs so spinners never fight)
    if chat_input_value and not chat.get("assistant_pending", False) and not chat.get(
        "drive_upload_pending", False
    ):
        if isinstance(chat_input_value, str):
            prompt = chat_input_value
            attached_files: list[Any] = []
        else:
            prompt = getattr(chat_input_value, "text", "") or ""
            attached_files = list(getattr(chat_input_value, "files", []) or [])
        video_file = attached_files[0] if attached_files else None
        user_text = prompt.strip() or "Sent an attachment."

        if not chat["messages"]:
            chat["title"] = (user_text[:48] + "…") if len(user_text) > 48 else user_text

        chat["messages"].append(
            {"role": "user", "content": user_text, "id": str(uuid.uuid4())}
        )

        fname: str | None = chat.get("video_name")
        if video_file is not None:
            raw = video_file.getvalue()
            mime = video_file.type or "video/mp4"
            fname = video_file.name or "upload.mp4"
            chat["video_bytes"] = raw
            chat["video_name"] = fname
            chat["video_mime"] = mime
            chat["video_seek_sec"] = 0
            chat["video_end_sec"] = None
            chat["video_seek_generation"] = int(chat.get("video_seek_generation", 0)) + 1
            chat["session_history"] = []
            chat["_turn_prompt"] = prompt
            chat["_turn_fname"] = fname
            chat["drive_upload_pending"] = True
            st.rerun()

        chat["_turn_prompt"] = prompt
        chat["_turn_fname"] = fname
        chat["assistant_pending"] = True
        st.rerun()

    # ── Layout ────────────────────────────────────────────────────────────────
    has_video = bool(chat.get("video_bytes"))

    if has_video:
        col_vid, col_chat = st.columns([5, 5], gap="large")
    else:
        col_vid = None
        col_chat = st.container()

    # LEFT: video only (draw before long LLM so the player stays visible while thinking)
    if has_video:
        with col_vid:
            with st.container(
                height="stretch",
                vertical_alignment="center",
                horizontal_alignment="center",
            ):
                st.subheader("Video")
                st.video(
                    io.BytesIO(chat["video_bytes"]),
                    format=chat.get("video_mime") or "video/mp4",
                    start_time=int(chat.get("video_seek_sec", 0)),
                )
                st.components.v1.html(SEEKBAR_CSS + autoplay_js, height=0)

    # RIGHT: chat + thinking spinner (spinner at bottom of this panel)
    with col_chat:
        st.title("Chat")
        st.caption("With a video + message, the reply includes playable ranges.")
        for msg in chat["messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="width:100%;margin:0.2rem 0 0.45rem 0;">'
                    f'<div style="margin-left:auto;width:fit-content;max-width:95%;'
                    f'background:linear-gradient(180deg,#2b3447 0%,#20283a 100%);'
                    f'border:1px solid rgba(255,255,255,0.13);border-radius:16px;'
                    f'padding:0.7rem 0.9rem;color:#f1f5f9;">'
                    f'{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                msg_id = _ensure_message_id(msg)
                content = msg.get("content") or ""
                if msg.get("timeframes") is not None:
                    render_reply_with_seek_links(content, msg_id, chat)
                    # if msg["timeframes"] and chat.get("video_bytes"):
                    #     with st.expander("Download clip segments"):
                    #         for idx, (a, b) in enumerate(msg["timeframes"], start=1):
                    #             clip_bytes = build_video_clip_bytes(
                    #                 chat["video_bytes"],
                    #                 int(a),
                    #                 int(b),
                    #                 chat.get("video_mime") or "video/mp4",
                    #             )
                    #             if clip_bytes is not None:
                    #                 st.download_button(
                    #                     label=f"Download [{a}:{b}]",
                    #                     data=clip_bytes,
                    #                     file_name=f"clip_{idx}_{a}_{b}.mp4",
                    #                     mime=chat.get("video_mime") or "video/mp4",
                    #                     key=f"dl_{st.session_state.active_chat_id}_{msg_id}_{idx}",
                    #                     use_container_width=True,
                    #                 )
                    #             else:
                    #                 st.caption(
                    #                     f"Clip [{a}:{b}] unavailable (ffmpeg missing)."
                    #                 )
                else:
                    st.markdown(content)
                if msg.get("drive_link"):
                    st.markdown(f"[Open on Google Drive]({msg['drive_link']})")

        # Drive upload on its own rerun so this spinner is never replaced by Thinking…
        if chat.get("drive_upload_pending", False) and chat.get("video_bytes"):
            with st.spinner("Analysing video ..."):
                up = upload_video_to_google_drive(
                    chat["video_bytes"],
                    filename=chat.get("video_name") or "upload.mp4",
                    mime_type=chat.get("video_mime") or "video/mp4",
                    folder_id=_drive_folder_id(),
                )
            chat["drive_upload_pending"] = False
            if not up.get("ok"):
                err = up.get("error", "Unknown error")
                reply = f"Drive upload failed: {err}\n\n"
                chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "drive_link": None,
                        "id": str(uuid.uuid4()),
                    }
                )
                st.rerun()
            chat["assistant_pending"] = True
            st.rerun()

        if chat.get("assistant_pending", False) and not chat.get(
            "drive_upload_pending", False
        ):
            with st.spinner("🧠 Thinking…"):
                reply, session_history = _invoke_assistant_model(chat)
                print("reply: ", reply)
            _append_assistant_turn_result(chat, reply, session_history)
            st.rerun()

    # ── Auto-scroll: inject directly into page head so it always runs ────────
    st.markdown(
        """
        <script>
        (function() {
            function scrollChat() {
                var col = document.querySelector(
                    '[data-testid="stHorizontalBlock"] > div:last-child'
                );
                if (col) {
                    col.scrollTop = col.scrollHeight - 120;
                    return;
                }
                // fallback when no video (single container)
                var main = document.querySelector('[data-testid="stMain"]');
                if (main) main.scrollTop = main.scrollHeight - 120;
            }
            // Run immediately and again after paint
            scrollChat();
            setTimeout(scrollChat, 100);
            setTimeout(scrollChat, 300);
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()



