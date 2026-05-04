import streamlit as st
from google import genai
from google.genai import types

KEYS = []

for name in ["GOOGLE_API_KEY"] + [f"GOOGLE_API_KEY_{i}" for i in range(1, 50)]:
    key = st.secrets.get(name)
    if key:
        KEYS.append(key)

def mask_key(key: str) -> str:
    return key[:8] + "..." + key[-4:]

for i, key in enumerate(KEYS, start=1):
    try:
        client = genai.Client(api_key=key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with OK only.",
            config=types.GenerateContentConfig(
                max_output_tokens=5,
                temperature=0,
            ),
        )

        print(f"✅ key_{i} {mask_key(key)} works: {response.text.strip()}")

    except Exception as e:
        err = str(e).lower()

        if "429" in err or "resource_exhausted" in err or "quota" in err or "rate" in err:
            print(f"❌ key_{i} {mask_key(key)} quota/rate limit exhausted")
        else:
            print(f"⚠️ key_{i} {mask_key(key)} other error: {e}")