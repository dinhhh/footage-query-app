"""Google Drive uploads using OAuth from `.streamlit/secrets.toml`.

Put OAuth data under `[google_drive]`:

- `credentials_json` — JSON string of the Google OAuth client (Desktop app), same as `credentials.json`.
- `token_json` — JSON string of the user token, same as `token.json`.

Optional: `folder_id` — default upload folder (see `app._drive_folder_id`).

Fallback files: `credentials.json` / `token.json` in the project root.

Run `python setup_drive_auth.py` once to open the browser and save the token into secrets.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tomllib
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ("https://www.googleapis.com/auth/drive.file",)

_DEFAULT_CREDS = Path(__file__).resolve().parent / "credentials.json"
_DEFAULT_TOKEN = Path(__file__).resolve().parent / "token.json"
_DEFAULT_TOKEN_CACHE = Path(__file__).resolve().parent / "drive_token_cache.json"
_DEFAULT_SECRETS = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"


def _credentials_path() -> Path:
    return Path(os.environ.get("GOOGLE_OAUTH_CREDENTIALS", _DEFAULT_CREDS))


def _token_path() -> Path:
    return Path(os.environ.get("GOOGLE_OAUTH_TOKEN", _DEFAULT_TOKEN))


def _token_cache_path() -> Path:
    return Path(os.environ.get("GOOGLE_OAUTH_TOKEN_CACHE", _DEFAULT_TOKEN_CACHE))


def _secrets_path() -> Path:
    env = os.environ.get("STREAMLIT_SECRETS_PATH")
    if env:
        return Path(env)
    return _DEFAULT_SECRETS


def _load_google_drive_block() -> dict[str, Any]:
    """Load `[google_drive]` from Streamlit secrets or `.streamlit/secrets.toml`."""
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            gd = st.secrets.get("google_drive")
            if gd is not None:
                return dict(gd)
    except Exception:
        pass

    path = _secrets_path()
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}
    block = data.get("google_drive")
    return dict(block) if isinstance(block, dict) else {}


def _parse_json_field(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def get_oauth_client_config() -> dict[str, Any] | None:
    """OAuth client JSON (`installed` or `web`) from secrets or credentials.json."""
    cfg = _parse_json_field(_load_google_drive_block().get("credentials_json"))
    if cfg is not None:
        return cfg
    creds_path = _credentials_path()
    if creds_path.is_file():
        try:
            with creds_path.open("r", encoding="utf-8") as f:
                parsed = json.load(f)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _persist_token_json(token_json_str: str) -> None:
    # Intentionally avoid writing refreshed tokens back to Streamlit secrets.
    cache_path = _token_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(token_json_str, encoding="utf-8")


def get_drive_credentials() -> Credentials | None:
    """Load/refresh OAuth credentials with one-time secrets bootstrap.

    Priority:
    1) `drive_token_cache.json`
    2) bootstrap `token_json` from Streamlit secrets (first run only), then persist to cache
    """
    token_json: dict[str, Any] | None = None

    cache_path = _token_cache_path()
    if cache_path.is_file():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                token_json = json.load(f)
        except Exception:
            token_json = None

    if token_json is None:
        block = _load_google_drive_block()
        token_json = _parse_json_field(block.get("token_json"))
        if token_json:
            _persist_token_json(json.dumps(token_json, separators=(",", ":")))

    if not token_json:
        return None

    creds = Credentials.from_authorized_user_info(token_json, SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        new_json = creds.to_json()
        _persist_token_json(new_json)
        _token_path().write_text(new_json, encoding="utf-8")
        return creds
    return None


def upload_video_to_google_drive(
    file_bytes: bytes,
    filename: str,
    mime_type: str = "video/mp4",
    folder_id: str | None = None,
) -> dict[str, Any]:
    """
    Upload a video file to the authenticated user's Google Drive.

    Args:
        file_bytes: Raw file content.
        filename: Name for the file on Drive.
        mime_type: MIME type (default video/mp4).
        folder_id: Optional Drive folder ID to upload into.

    Returns:
        dict with keys: ok (bool), file_id, web_view_link, web_content_link, name, error (optional).
    """
    creds = get_drive_credentials()
    if creds is None:
        return {
            "ok": False,
            "error": (
                "Drive not authorized. Add [google_drive] credentials_json and token_json "
                "to .streamlit/secrets.toml, or run `python setup_drive_auth.py`, then retry."
            ),
        }

    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        file_metadata: dict[str, Any] = {"name": filename}
        if folder_id:
            file_metadata["parents"] = [folder_id]

        media = MediaIoBaseUpload(
            io.BytesIO(file_bytes),
            mimetype=mime_type,
            resumable=True,
        )
        created = (
            service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id, name, mimeType, webViewLink, webContentLink",
                supportsAllDrives=True,
            )
            .execute()
        )

        return {
            "ok": True,
            "file_id": created.get("id"),
            "name": created.get("name"),
            "web_view_link": created.get("webViewLink"),
            "web_content_link": created.get("webContentLink"),
        }
    except Exception as e:  # noqa: BLE001 — surface any API error to the UI
        return {"ok": False, "error": str(e)}
