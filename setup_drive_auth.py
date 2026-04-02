"""One-time OAuth setup: saves token to local cache/token files."""

from google_auth_oauthlib.flow import InstalledAppFlow

from drive_upload import (
    SCOPES,
    get_oauth_client_config,
    _persist_token_json,
    _token_path,
)


def main() -> None:
    client_config = get_oauth_client_config()
    if not client_config:
        raise SystemExit(
            "Missing OAuth client. Add [google_drive].credentials_json to .streamlit/secrets.toml "
            "(JSON string of your Desktop OAuth client), or place credentials.json in this folder."
        )
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0)
    token_json = creds.to_json()
    _persist_token_json(token_json)
    _token_path().write_text(token_json, encoding="utf-8")
    print("Saved token to local cache/token files (no write to Streamlit secrets).")


if __name__ == "__main__":
    main()
