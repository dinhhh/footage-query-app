"""
Streamlit chat UI (ChatGPT-style) with random replies, optional Google Drive video
upload, and clickable time ranges to seek the in-session video.
Run: streamlit run app.py
"""

from __future__ import annotations

import io
import mimetypes
import random
import subprocess
import tempfile
import uuid
from typing import Any

import streamlit as st

from drive_upload import upload_video_to_google_drive

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


def render_play_button(a: int, b: int, msg_id: str) -> None:
    cid = st.session_state.active_chat_id
    st.button(
        f"\u25b6 [{a}:{b}]",
        key=f"play_{cid}_{msg_id}",
        on_click=_play_on_click,
        kwargs={"a": a, "b": b},
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
        return st.secrets.get("GOOGLE_DRIVE_FOLDER_ID", None)
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
    }


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
                padding-bottom: 140px !important;
                scroll-behavior: smooth !important;
            }
            /* LEFT video column: no scroll, sticks to top */
            [data-testid="stHorizontalBlock"] > div:first-child {
                position: sticky !important;
                top: 0 !important;
                align-self: flex-start !important;
                max-height: calc(100vh - 140px) !important;
                overflow: hidden !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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

    # ── Layout ────────────────────────────────────────────────────────────────
    has_video = bool(chat.get("video_bytes"))

    if has_video:
        col_vid, col_chat = st.columns([5, 5], gap="large")
    else:
        col_vid  = None
        col_chat = st.container()

    # LEFT: video panel
    if has_video:
        with col_vid:
            st.subheader("Video")
            st.video(
                io.BytesIO(chat["video_bytes"]),
                format=chat.get("video_mime") or "video/mp4",
                start_time=int(chat.get("video_seek_sec", 0)),
            )
            st.components.v1.html(SEEKBAR_CSS + autoplay_js, height=0)

    # RIGHT: chat panel
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
                st.markdown(msg["content"])
                if msg.get("timeframes"):
                    msg_id = _ensure_message_id(msg)
                    for idx, (a, b) in enumerate(msg["timeframes"][:5], start=1):
                        c1, c2 = st.columns(2)
                        with c1:
                            render_play_button(a, b, f"{msg_id}_{idx}")
                        with c2:
                            if chat.get("video_bytes"):
                                clip_bytes = build_video_clip_bytes(
                                    chat["video_bytes"], int(a), int(b),
                                    chat.get("video_mime") or "video/mp4",
                                )
                                if clip_bytes is not None:
                                    st.download_button(
                                        label=f"Download [{a}:{b}]",
                                        data=clip_bytes,
                                        file_name=f"clip_{idx}_{a}_{b}.mp4",
                                        mime=chat.get("video_mime") or "video/mp4",
                                        key=f"dl_{st.session_state.active_chat_id}_{msg_id}_{idx}",
                                        use_container_width=True,
                                    )
                                else:
                                    st.caption("Clip download unavailable (ffmpeg missing).")
                if msg.get("drive_link"):
                    st.markdown(f"[Open on Google Drive]({msg['drive_link']})")

        st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)
    # ── Auto-scroll: reliably run inside an HTML component ───────────────────
    st.components.v1.html(
        """
        <script>
        (function() {
            function autoScroll() {
                const parentDoc = window.parent.document;

                const chatCol = parentDoc.querySelector(
                    '[data-testid="stHorizontalBlock"] > div:last-child'
                );

                const anchor = parentDoc.getElementById("chat-bottom-anchor");

                if (chatCol && anchor) {
                    const anchorTop = anchor.getBoundingClientRect().top;
                    const chatTop = chatCol.getBoundingClientRect().top;
                    const target = chatCol.scrollTop + (anchorTop - chatTop) - 180;
                    chatCol.scrollTo({
                        top: Math.max(target, 0),
                        behavior: "smooth"
                    });
                    return;
                }

                if (chatCol) {
                    chatCol.scrollTo({
                        top: Math.max(chatCol.scrollHeight - chatCol.clientHeight - 180, 0),
                        behavior: "smooth"
                    });
                    return;
                }

                const main = parentDoc.querySelector('[data-testid="stMain"]');
                if (main) {
                    main.scrollTo({
                        top: Math.max(main.scrollHeight - main.clientHeight - 180, 0),
                        behavior: "smooth"
                    });
                }
            }

            setTimeout(autoScroll, 50);
            setTimeout(autoScroll, 200);
            setTimeout(autoScroll, 500);
        })();
        </script>
        """,
        height=0,
    )

    # ── Handle input ──────────────────────────────────────────────────────────
    if chat_input_value:
        if isinstance(chat_input_value, str):
            prompt = chat_input_value
            attached_files = []
        else:
            prompt = getattr(chat_input_value, "text", "") or ""
            attached_files = list(getattr(chat_input_value, "files", []) or [])

        video_file = attached_files[0] if attached_files else None
        user_text  = prompt.strip() or "Sent an attachment."

        if not chat["messages"]:
            chat["title"] = (user_text[:48] + "…") if len(user_text) > 48 else user_text
        chat["messages"].append({"role": "user", "content": user_text, "id": str(uuid.uuid4())})

        if video_file is not None:
            raw   = video_file.getvalue()
            mime  = video_file.type or "video/mp4"
            fname = video_file.name or "upload.mp4"
            chat["video_bytes"] = raw
            chat["video_name"]  = fname
            chat["video_mime"]  = mime
            chat["video_seek_sec"] = 0
            chat["video_end_sec"]  = None
            chat["video_seek_generation"] = int(chat.get("video_seek_generation", 0)) + 1

            with st.spinner("Uploading and processing video..."):
                up = upload_video_to_google_drive(raw, filename=fname, mime_type=mime,
                                                   folder_id=_drive_folder_id())

            timeframes = generate_random_timeframes(max_clips=5)
            tf_label   = ", ".join([f"[{a}:{b}]" for a, b in timeframes])
            base       = generate_random_text_response(prompt)

            if up.get("ok"):
                link = up.get("web_view_link") or up.get("web_content_link") or ""
                if base == NOT_IN_VIDEO_RESPONSE:
                    chat["messages"].append({"role": "assistant", "content": f"{base}\n\nVideo uploaded to your Google Drive.", "drive_link": link, "id": str(uuid.uuid4())})
                else:
                    chat["messages"].append({"role": "assistant", "content": f"{base}\n\nVideo uploaded to your Google Drive. Suggested segments (seconds) **{tf_label}** — use the buttons below to seek.", "timeframes": timeframes, "drive_link": link, "id": str(uuid.uuid4())})
            else:
                err = up.get("error", "Unknown error")
                if base == NOT_IN_VIDEO_RESPONSE:
                    chat["messages"].append({"role": "assistant", "content": f"{base}\n\nDrive upload failed: {err}", "drive_link": None, "id": str(uuid.uuid4())})
                else:
                    chat["messages"].append({"role": "assistant", "content": f"{base}\n\nDrive upload failed: {err}\n\nStill suggesting segments **{tf_label}** for the video in this chat.", "timeframes": timeframes, "drive_link": None, "id": str(uuid.uuid4())})
        else:
            with st.spinner("Analyzing video and retrieving clips..."):
                reply      = generate_random_text_response(prompt)
                timeframes = generate_random_timeframes(max_clips=5)
            tf_label = ", ".join([f"[{a}:{b}]" for a, b in timeframes])
            if reply == NOT_IN_VIDEO_RESPONSE:
                chat["messages"].append({"role": "assistant", "content": reply, "id": str(uuid.uuid4())})
            else:
                chat["messages"].append({"role": "assistant", "content": f"{reply}\n\nSuggested segments (seconds) **{tf_label}**.", "timeframes": timeframes, "id": str(uuid.uuid4())})

        st.rerun()


if __name__ == "__main__":
    main()
