"""One-time OAuth setup: saves token into `.streamlit/secrets.toml` under [google_drive]."""

import json

from google_auth_oauthlib.flow import InstalledAppFlow

from drive_upload import (
    SCOPES,
    get_oauth_client_config,
    merge_google_drive_secrets,
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
    creds_json = json.dumps(client_config, separators=(",", ":"))
    merge_google_drive_secrets(
        credentials_json_str=creds_json,
        token_json_str=token_json,
    )
    print(f"Saved credentials and token to .streamlit/secrets.toml under [google_drive].")


if __name__ == "__main__":
    main()
