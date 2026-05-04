# ============================================================
# IMPORTS
# ============================================================

import os
import time
from typing import Any, List, Optional

from google import genai


# ============================================================
# ERROR DETECTION
# ============================================================

QUOTA_ERROR_PATTERNS = [
    "resource_exhausted",
    "quota",
    "quota exceeded",
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "requests per minute",
    "tokens per minute",
    "free tier",
]


TEMPORARY_MODEL_ERROR_PATTERNS = [
    # Gemini high demand / unavailable
    "503",
    "unavailable",
    "model is currently experiencing high demand",
    "spikes in demand",
    "try again later",
    "temporarily unavailable",
    "temporary",

    # Timeout / deadline / server-side temporary errors
    "timeout",
    "timed out",
    "read timed out",
    "deadline",
    "deadline exceeded",
    "504",
    "gateway timeout",
    "internal",
    "500",
    "server error",
]


def is_quota_or_rate_error(error: Any) -> bool:
    """
    Detect whether an exception or text looks like a quota/rate-limit error.

    If this returns True, switching API key can help.
    """
    text = str(error or "").lower()

    return any(pattern in text for pattern in QUOTA_ERROR_PATTERNS)


def is_temporary_model_error(error: Any) -> bool:
    """
    Detect whether Gemini is temporarily unavailable, overloaded, or timed out.

    If this returns True:
    - retry the same key first
    - if same key still fails after retries, switch to the next key
    """
    text = str(error or "").lower()

    return any(pattern in text for pattern in TEMPORARY_MODEL_ERROR_PATTERNS)


# ============================================================
# SECRET LOADING HELPERS
# ============================================================

def _safe_get(mapping: Any, key: str, default: Any = None) -> Any:
    """
    Safely read value from dict-like objects such as os.environ or st.secrets.
    """
    try:
        return mapping.get(key, default)
    except Exception:
        return default


def _append_key(keys: List[str], value: Any) -> None:
    """
    Append a valid key string.
    """
    if not isinstance(value, str):
        return

    value = value.strip()

    if value:
        keys.append(value)


def _load_streamlit_secrets() -> Any:
    """
    Load Streamlit secrets if available.

    This supports .streamlit/secrets.toml.
    """
    try:
        import streamlit as st

        return st.secrets

    except Exception:
        return {}


def _dedupe_keep_order(values: List[str]) -> List[str]:
    """
    Remove duplicate API keys while preserving order.
    """
    seen = set()
    result = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def load_gemini_api_keys(max_numbered_keys: int = 100) -> List[str]:
    """
    Load Gemini / Google API keys from environment variables and Streamlit secrets.

    Supported formats:

    Root-level .streamlit/secrets.toml:
        GOOGLE_API_KEY="..."
        GOOGLE_API_KEY_1="..."
        GOOGLE_API_KEY_2="..."

    Also supported:
        GEMINI_API_KEY="..."
        GEMINI_API_KEY_1="..."

    Nested format:
        [gemini]
        api_key="..."
        api_keys=["key1", "key2"]
        api_key_1="..."
    """
    keys: List[str] = []

    secrets = _load_streamlit_secrets()

    root_names = ["GOOGLE_API_KEY", "GEMINI_API_KEY"]

    for idx in range(1, max_numbered_keys + 1):
        root_names.append(f"GOOGLE_API_KEY_{idx}")
        root_names.append(f"GEMINI_API_KEY_{idx}")

    for source in [os.environ, secrets]:
        for name in root_names:
            _append_key(keys, _safe_get(source, name))

    gemini_section = _safe_get(secrets, "gemini", {})

    if gemini_section:
        _append_key(keys, _safe_get(gemini_section, "api_key"))

        api_keys = _safe_get(gemini_section, "api_keys", [])

        if isinstance(api_keys, (list, tuple)):
            for value in api_keys:
                _append_key(keys, value)

        for idx in range(1, max_numbered_keys + 1):
            _append_key(keys, _safe_get(gemini_section, f"api_key_{idx}"))

    return _dedupe_keep_order(keys)


# ============================================================
# GEMINI KEY MANAGER
# ============================================================

class GeminiKeyManager:
    """
    Manages multiple Gemini API keys.

    Rules:
    - 429 / quota / rate limit:
        switch to next key immediately.

    - 503 / timeout / high demand:
        reset current client
        sleep
        retry same key up to temporary_error_retries times
        if still failed, switch to next key

    - Other errors:
        raise directly.
    """

    def __init__(
        self,
        api_keys: List[str],
        sleep_after_rotation: float = 1.0,
        temporary_error_retries: int = 4,
        temporary_error_sleep_seconds: float = 30.0,
    ):
        if not api_keys:
            raise ValueError(
                "No Gemini API key found. Add GOOGLE_API_KEY or GOOGLE_API_KEY_1 "
                "to .streamlit/secrets.toml or environment variables."
            )

        self.api_keys = api_keys
        self.index = 0
        self.sleep_after_rotation = sleep_after_rotation
        self.temporary_error_retries = temporary_error_retries
        self.temporary_error_sleep_seconds = temporary_error_sleep_seconds
        self._client = None

        self.activate_current_key()

    @classmethod
    def from_sources(cls) -> "GeminiKeyManager":
        """
        Create a key manager by loading all keys from secrets/environment.
        """
        return cls(api_keys=load_gemini_api_keys())

    @property
    def key_count(self) -> int:
        return len(self.api_keys)

    @property
    def current_key(self) -> str:
        return self.api_keys[self.index]

    def current_key_label(self) -> str:
        """
        Safe label that does not reveal the actual key.
        """
        return f"key_{self.index + 1}/{self.key_count}"

    def activate_current_key(self) -> None:
        """
        Put current key into environment variables used by google.genai.
        """
        os.environ["GOOGLE_API_KEY"] = self.current_key
        os.environ["GEMINI_API_KEY"] = self.current_key

    def reset_current_client(self) -> None:
        """
        Reset the current client without switching API key.

        Use this for 503 / timeout / high demand retries.
        """
        self._client = None
        self.activate_current_key()

        print(
            f"[Gemini temporary error] Reset client for "
            f"{self.current_key_label()} without switching key."
        )

    @property
    def current_client(self):
        """
        Lazily create a google.genai Client for the active key.
        """
        if self._client is None:
            self.activate_current_key()
            self._client = genai.Client(api_key=self.current_key)

        return self._client

    def switch_to_next_key(self) -> bool:
        """
        Switch to the next API key.

        Returns:
            True if switched successfully.
            False if there are no keys left.
        """
        if self.index + 1 >= self.key_count:
            return False

        self.index += 1
        self._client = None
        self.activate_current_key()

        print(f"[Gemini key rotation] Switched to {self.current_key_label()}.")

        return True

    def build_client_proxy(self):
        """
        Return a proxy client that supports:
            client.models.generate_content(...)
            client.models.embed_content(...)

        This avoids editing video_pipeline.py and llm_pipeline.py.
        """
        return RotatingGeminiClientProxy(self)

    def patch_modules(self, *modules: Any) -> List[str]:
        """
        Replace module-level `client` variables with the rotating proxy.
        """
        proxy = self.build_client_proxy()
        patched = []

        for module in modules:
            if module is None:
                continue

            if hasattr(module, "client"):
                setattr(module, "client", proxy)
                patched.append(getattr(module, "__name__", str(module)))

        return patched


# ============================================================
# GEMINI CLIENT PROXY
# ============================================================

class RotatingGeminiClientProxy:
    """
    Proxy object that looks like genai.Client for the parts used in the project.
    """

    def __init__(self, key_manager: GeminiKeyManager):
        self.key_manager = key_manager

    @property
    def models(self):
        return RotatingGeminiModelsProxy(self.key_manager)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.key_manager.current_client, name)


class RotatingGeminiModelsProxy:
    """
    Proxy for client.models.
    """

    def __init__(self, key_manager: GeminiKeyManager):
        self.key_manager = key_manager

    def __getattr__(self, method_name: str) -> Any:
        target = getattr(self.key_manager.current_client.models, method_name)

        if not callable(target):
            return target

        def call_with_key_rotation(*args, **kwargs):
            last_error: Optional[Exception] = None

            while True:
                try:
                    method = getattr(
                        self.key_manager.current_client.models,
                        method_name,
                    )

                    return method(*args, **kwargs)

                except Exception as e:
                    last_error = e

                    # ====================================================
                    # 1. Quota / rate limit / 429
                    # ====================================================
                    # Switch key immediately.
                    if is_quota_or_rate_error(e):
                        print(
                            f"[Gemini key rotation] {method_name} quota/rate error on "
                            f"{self.key_manager.current_key_label()}: {str(e)[:300]}"
                        )

                        switched = self.key_manager.switch_to_next_key()

                        if not switched:
                            print(
                                "[Gemini key rotation] All API keys are quota/rate limited "
                                "or exhausted. No more keys to try."
                            )
                            raise last_error

                        print(
                            f"[Gemini key rotation] Retrying {method_name} with "
                            f"{self.key_manager.current_key_label()}..."
                        )

                        time.sleep(self.key_manager.sleep_after_rotation)
                        continue

                    # ====================================================
                    # 2. Temporary model / timeout / 503 errors
                    # ====================================================
                    # Retry same key first.
                    # If same key still fails after N retries, switch key.
                    if is_temporary_model_error(e):
                        print(
                            f"[Gemini temporary error] {method_name} failed on "
                            f"{self.key_manager.current_key_label()}: {str(e)[:300]}"
                        )
                        print(
                            "[Gemini temporary error] 503/timeout/high-demand detected. "
                            "Retrying SAME key first. Will switch key only if retries fail."
                        )

                        for attempt in range(self.key_manager.temporary_error_retries):
                            self.key_manager.reset_current_client()

                            print(
                                f"[Gemini temporary error] Sleeping "
                                f"{self.key_manager.temporary_error_sleep_seconds:.1f}s "
                                f"before retry same key "
                                f"{attempt + 1}/{self.key_manager.temporary_error_retries}..."
                            )

                            time.sleep(self.key_manager.temporary_error_sleep_seconds)

                            try:
                                method = getattr(
                                    self.key_manager.current_client.models,
                                    method_name,
                                )

                                return method(*args, **kwargs)

                            except Exception as retry_error:
                                last_error = retry_error

                                # If retry becomes quota error, switch key immediately.
                                if is_quota_or_rate_error(retry_error):
                                    print(
                                        f"[Gemini key rotation] Retry became quota/rate error on "
                                        f"{self.key_manager.current_key_label()}: "
                                        f"{str(retry_error)[:300]}"
                                    )

                                    switched = self.key_manager.switch_to_next_key()

                                    if not switched:
                                        print(
                                            "[Gemini key rotation] All API keys are quota/rate "
                                            "limited or exhausted. No more keys to try."
                                        )
                                        raise last_error

                                    print(
                                        f"[Gemini key rotation] Retrying {method_name} with "
                                        f"{self.key_manager.current_key_label()}..."
                                    )

                                    time.sleep(self.key_manager.sleep_after_rotation)
                                    break

                                # Still temporary: keep retrying same key.
                                if is_temporary_model_error(retry_error):
                                    print(
                                        f"[Gemini temporary error] Still temporary error on "
                                        f"{self.key_manager.current_key_label()}: "
                                        f"{str(retry_error)[:300]}"
                                    )
                                    continue

                                # Other errors should not be hidden.
                                raise retry_error

                        else:
                            # This happens when all same-key temporary retries fail.
                            # Now switch to next key.
                            print(
                                "[Gemini temporary error] Temporary error persisted after "
                                f"{self.key_manager.temporary_error_retries} same-key retries. "
                                "Switching to next key now."
                            )

                            switched = self.key_manager.switch_to_next_key()

                            if not switched:
                                print(
                                    "[Gemini key rotation] Temporary error persisted and "
                                    "no more keys are available. No more keys to try."
                                )
                                raise last_error

                            print(
                                f"[Gemini key rotation] Retrying {method_name} with "
                                f"{self.key_manager.current_key_label()} after temporary failures..."
                            )

                            time.sleep(self.key_manager.sleep_after_rotation)
                            continue

                        # Reached only if retry became quota error and key switched.
                        continue

                    # ====================================================
                    # 3. Other errors
                    # ====================================================
                    # Do not rotate key because switching key probably will not help.
                    raise e

        return call_with_key_rotation