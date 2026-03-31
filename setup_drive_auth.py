"""One-time OAuth setup: creates token.json for Google Drive uploads."""

from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

from drive_upload import SCOPES, _credentials_path, _token_path


def main() -> None:
    creds_path = _credentials_path()
    token_path = _token_path()
    if not creds_path.is_file():
        raise SystemExit(
            f"Missing {creds_path}. Download OAuth client JSON from Google Cloud Console "
            "(Desktop app) and save it as credentials.json in this folder."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    creds = flow.run_local_server(port=0)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"Saved tokens to {token_path}")


if __name__ == "__main__":
    main()
