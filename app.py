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

# --- Response generators (wrapped as requested) ---
NOT_IN_VIDEO_RESPONSE = "This content does not exist in your video"


def _jump_on_click(**kwargs: Any) -> None:
    """Use **kwargs (not args=) so each button keeps its own a,b — args= can break with multiple buttons."""
    a = int(kwargs["a"])
    b = int(kwargs["b"])
    cid = st.session_state.active_chat_id
    st.session_state[f"seek_range_{cid}"] = (a, b)


def _ensure_message_id(msg: dict[str, Any]) -> str:
    mid = msg.get("id")
    if mid is None:
        mid = str(uuid.uuid4())
        msg["id"] = mid
    return mid


def render_jump_button(a: int, b: int, msg_id: str) -> None:
    cid = st.session_state.active_chat_id
    label = f"Jump to [{a}; {b}]"
    st.button(
        label,
        key=f"jump_{cid}_{msg_id}",
        on_click=_jump_on_click,
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
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),
            "-to",
            str(end_sec),
            "-i",
            in_path,
            "-c",
            "copy",
            out_path,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(out_path, "rb") as f:
                return f.read()
        except Exception:
            return None


def generate_random_text_response(user_message: str) -> str:
    """Server-side placeholder: returns random canned text (no LLM)."""
    templates = [
        "Interesting point about: “{msg}”. Here is a random thought: the weather is computational.",
        "You said “{msg}”. I’ll reply with something arbitrary: seven platypuses in a trench coat.",
        "Acknowledged: {msg}. Random line: static on the radio, but friendly.",
        "On “{msg}”: a random sentence — the elevator music was written in binary.",
        "Re: {msg} — imagine a cloud made of toast. That’s the vibe.",
        NOT_IN_VIDEO_RESPONSE,
    ]
    selected = random.choice(templates)
    if selected == NOT_IN_VIDEO_RESPONSE:
        return selected
    return selected.format(msg=user_message.strip() or "(empty message)")


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


def _drive_folder_id() -> str | None:
    try:
        return st.secrets.get("GOOGLE_DRIVE_FOLDER_ID", None)
    except Exception:
        return None


def _init_session() -> None:
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "chat_order" not in st.session_state:
        st.session_state.chat_order = []
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat_id" not in st.session_state:
        cid = str(uuid.uuid4())
        st.session_state.chats[cid] = {
            "title": "New chat",
            "messages": [],
            "video_bytes": None,
            "video_name": None,
            "video_mime": "video/mp4",
            "video_seek_sec": 0,
            "video_end_sec": None,
            "video_seek_generation": 0,
        }
        st.session_state.chat_order.append(cid)
        st.session_state.active_chat_id = cid


def _active_chat() -> dict[str, Any]:
    return st.session_state.chats[st.session_state.active_chat_id]


def _new_chat() -> None:
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = {
        "title": "New chat",
        "messages": [],
        "video_bytes": None,
        "video_name": None,
        "video_mime": "video/mp4",
        "video_seek_sec": 0,
        "video_end_sec": None,
        "video_seek_generation": 0,
    }
    st.session_state.chat_order.append(cid)
    st.session_state.active_chat_id = cid


def _delete_current_chat() -> None:
    cid = st.session_state.active_chat_id
    st.session_state.chat_order = [x for x in st.session_state.chat_order if x != cid]
    del st.session_state.chats[cid]
    if not st.session_state.chat_order:
        _new_chat()
    else:
        st.session_state.active_chat_id = st.session_state.chat_order[-1]


def _apply_chatgpt_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.2rem; max-width: 900px; }
            div[data-testid="stChatMessage"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Chat", page_icon="💬", layout="centered")
    _apply_chatgpt_css()
    _init_session()

    with st.sidebar:
        if st.button("➕ New chat", use_container_width=True):
            _new_chat()
            st.rerun()
        if st.button("🗑️ Delete current chat", use_container_width=True):
            _delete_current_chat()
            st.rerun()

        with st.expander("Your chats", expanded=True):
            for cid in reversed(st.session_state.chat_order):
                chat = st.session_state.chats[cid]
                label = chat["title"][:36] + ("…" if len(chat["title"]) > 36 else "")
                if st.button(label, key=f"tab_{cid}", use_container_width=True):
                    st.session_state.active_chat_id = cid
                    st.rerun()

        st.divider()
        st.caption("Attach a video, then send a message to upload to Drive and get a time range.")
        video_file = st.file_uploader(
            "Video file",
            type=["mp4", "webm", "mov", "mkv", "avi"],
            key=f"sidebar_video_{st.session_state.uploader_key}",
        )

    chat = _active_chat()

    st.title("Chat")
    st.caption("Messages get random placeholder replies. With a video + message, the reply includes a jumpable range.")

    range_key = f"seek_range_{st.session_state.active_chat_id}"
    if range_key in st.session_state:
        a, b = st.session_state.pop(range_key)
        a, b = int(a), int(b)
        if b <= a:
            b = a + 1
        chat["video_seek_sec"] = a
        chat["video_end_sec"] = b
        chat["video_seek_generation"] = int(chat.get("video_seek_generation", 0)) + 1

    video_slot = st.empty()
    if chat.get("video_bytes"):
        st.subheader("Video in this chat")
        end_s = chat.get("video_end_sec")
        with video_slot.container():
            st.video(
                io.BytesIO(chat["video_bytes"]),
                format=chat.get("video_mime") or "video/mp4",
                start_time=int(chat.get("video_seek_sec", 0)),
                end_time=int(end_s) if end_s is not None else None,
            )

    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("timeframe"):
                a, b = msg["timeframe"]
                msg_id = _ensure_message_id(msg)
                c1, c2 = st.columns(2)
                with c1:
                    render_jump_button(a, b, msg_id)
                with c2:
                    if chat.get("video_bytes"):
                        clip_bytes = build_video_clip_bytes(
                            chat["video_bytes"],
                            int(a),
                            int(b),
                            chat.get("video_mime") or "video/mp4",
                        )
                        if clip_bytes is not None:
                            st.download_button(
                                label=f"Download [{a}; {b}]",
                                data=clip_bytes,
                                file_name=f"clip_{a}_{b}.mp4",
                                mime=chat.get("video_mime") or "video/mp4",
                                key=f"dl_{st.session_state.active_chat_id}_{msg_id}",
                                use_container_width=True,
                            )
                        else:
                            st.caption("Clip download unavailable (ffmpeg missing).")
            if msg["role"] == "assistant" and msg.get("drive_link"):
                st.markdown(f"[Open on Google Drive]({msg['drive_link']})")

    if prompt := st.chat_input("Message…"):
        if not chat["messages"]:
            chat["title"] = (prompt[:48] + "…") if len(prompt) > 48 else prompt

        chat["messages"].append(
            {"role": "user", "content": prompt, "id": str(uuid.uuid4())}
        )

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

            up = upload_video_to_google_drive(
                raw,
                filename=fname,
                mime_type=mime,
                folder_id=_drive_folder_id(),
            )
            t0, t1 = generate_random_timeframe_seconds()
            tf_label = f"[{t0}; {t1}]"
            base = generate_random_text_response(prompt)
            if up.get("ok"):
                link = up.get("web_view_link") or up.get("web_content_link") or ""
                if base == NOT_IN_VIDEO_RESPONSE:
                    reply = f"{base}\n\nVideo uploaded to your Google Drive."
                    chat["messages"].append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "drive_link": link,
                            "id": str(uuid.uuid4()),
                        }
                    )
                else:
                    reply = (
                        f"{base}\n\n"
                        f"Video uploaded to your Google Drive. "
                        f"Suggested segment (seconds) **{tf_label}** — use the button below to seek."
                    )
                    chat["messages"].append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "timeframe": (t0, t1),
                            "drive_link": link,
                            "id": str(uuid.uuid4()),
                        }
                    )
            else:
                err = up.get("error", "Unknown error")
                if base == NOT_IN_VIDEO_RESPONSE:
                    reply = f"{base}\n\nDrive upload failed: {err}"
                    chat["messages"].append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "drive_link": None,
                            "id": str(uuid.uuid4()),
                        }
                    )
                else:
                    reply = (
                        f"{base}\n\n"
                        f"Drive upload failed: {err}\n\n"
                        f"Still suggesting segment **{tf_label}** for the video in this chat."
                    )
                    chat["messages"].append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "timeframe": (t0, t1),
                            "drive_link": None,
                            "id": str(uuid.uuid4()),
                        }
                    )
            st.session_state.uploader_key += 1
        else:
            reply = generate_random_text_response(prompt)
            t0, t1 = generate_random_timeframe_seconds()
            tf_label = f"[{t0}; {t1}]"
            if reply == NOT_IN_VIDEO_RESPONSE:
                chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "id": str(uuid.uuid4()),
                    }
                )
            else:
                reply_with_tf = f"{reply}\n\nSuggested segment (seconds) **{tf_label}**."
                chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": reply_with_tf,
                        "timeframe": (t0, t1),
                        "id": str(uuid.uuid4()),
                    }
                )

        st.rerun()


if __name__ == "__main__":
    main()
