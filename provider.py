"""Gemini HTTP transport. Exceptions carry only safe diagnostic codes."""
import os
import re

import requests

MODEL = os.getenv("DOCUMIND_MODEL", "gemini-3.5-flash-lite")


class ProviderError(Exception):
    pass


def generate(parts, key, temperature=0.7, *, max_tokens=1600, timeout=75):
    if not re.fullmatch(r"[\w.\-]+", MODEL):
        raise ValueError("Invalid model ID")
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            headers={"x-goog-api-key": key},
            json={"contents": [{"role": "user", "parts": parts}],
                  "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}},
            timeout=(8, timeout),
        )
    except requests.RequestException:
        raise ProviderError("provider_unreachable") from None
    if not response.ok:
        raise ProviderError(f"provider_http_{response.status_code}")
    try:
        parts = response.json()["candidates"][0]["content"]["parts"]
        answer = "".join(part.get("text", "") for part in parts if not part.get("thought")).strip()
    except (ValueError, KeyError, IndexError, TypeError):
        answer = ""
    if not answer:
        raise ProviderError("provider_empty_response")
    return answer
