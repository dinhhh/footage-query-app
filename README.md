# iLab Streamlit Chat + Google Drive Upload

This project is a Streamlit chat UI that can:
- upload video files to your Google Drive,
- suggest clickable time ranges in chat,
- cut and download video clips using `ffmpeg`.

## 1) Install FFmpeg on your system

The app uses `ffmpeg` for clip extraction. Install it so the `ffmpeg` command works in terminal.

### macOS (Homebrew)
```bash
brew install ffmpeg
ffmpeg -version
```

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -version
```

### Windows (winget)
```powershell
winget install Gyan.FFmpeg
ffmpeg -version
```

## 2) Create Python environment and install dependencies

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Get Google Drive credentials from Google Cloud

This app needs **OAuth client credentials** (not only a simple API key), because it uploads to a user's Drive.

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create/select a project.
3. Enable **Google Drive API**:
   - APIs & Services -> Library -> search "Google Drive API" -> Enable.
4. Configure OAuth consent screen:
   - APIs & Services -> OAuth consent screen.
   - Choose External/Internal, fill required fields, add yourself as a test user if needed.
5. Create OAuth client credentials:
   - APIs & Services -> Credentials -> Create Credentials -> **OAuth client ID**.
   - Application type: **Desktop app**.
   - Download the JSON file.

## 4) Put credentials in `.streamlit/secrets.toml` (your format)

Create/edit `.streamlit/secrets.toml` and keep this exact structure:

```toml
[google_drive]
credentials_json = "{\"installed\":{\"client_id\":\"YOUR_CLIENT_ID.apps.googleusercontent.com\",\"project_id\":\"YOUR_PROJECT_ID\",\"auth_uri\":\"https://accounts.google.com/o/oauth2/auth\",\"token_uri\":\"https://oauth2.googleapis.com/token\",\"auth_provider_x509_cert_url\":\"https://www.googleapis.com/oauth2/v1/certs\",\"client_secret\":\"YOUR_CLIENT_SECRET\",\"redirect_uris\":[\"http://localhost\"]}}"
token_json = ""
```

Notes:
- `credentials_json` is the downloaded OAuth client JSON converted to a single JSON string.
- Leave `token_json` empty first. It will be generated after OAuth login.
- Optional:
  - `folder_id = "YOUR_DRIVE_FOLDER_ID"` under `[google_drive]` to upload into a specific folder.

## 5) Generate `token_json` automatically

Run one-time auth flow:

```bash
python setup_drive_auth.py
```

This opens a browser, asks for Google permission, then saves both credentials and token into `.streamlit/secrets.toml` under `[google_drive]`.

## 6) Run the app

```bash
streamlit run app.py
```

Open the local Streamlit URL, upload a video, and send a message to test Drive upload + timeframe features.
