"""Upload videos to the signed-in user's Google Drive (OAuth 2.0).

Place a Google Cloud OAuth *Desktop app* client JSON next to this project as
`credentials.json`. Run `python setup_drive_auth.py` once in a terminal to open
the browser and create `token.json`.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ("https://www.googleapis.com/auth/drive.file",)

_DEFAULT_CREDS = Path(__file__).resolve().parent / "credentials.json"
_DEFAULT_TOKEN = Path(__file__).resolve().parent / "token.json"


def _credentials_path() -> Path:
    return Path(os.environ.get("GOOGLE_OAUTH_CREDENTIALS", _DEFAULT_CREDS))


def _token_path() -> Path:
    return Path(os.environ.get("GOOGLE_OAUTH_TOKEN", _DEFAULT_TOKEN))


def get_drive_credentials() -> Credentials | None:
    """Load or refresh OAuth credentials; returns None if token file is missing."""
    token_path = _token_path()
    creds_path = _credentials_path()
    if not token_path.is_file():
        return None
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
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
                "Drive not authorized. Add credentials.json, run "
                "`python setup_drive_auth.py` once, then retry."
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
