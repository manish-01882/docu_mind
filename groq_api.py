"""Groq chat-completions wrapper with retry handling and an offline fallback."""

from __future__ import annotations

import os
import random
import re
import time

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return False


load_dotenv()
API_URL = "https://api.groq.com/openai/v1/chat/completions"
TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.6-27b"

# Free-tier accounts allow roughly 30 requests per minute, and indexing a PDF
# issues one request per extracted element, so bursts hit 429 routinely.
DEFAULT_MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 60.0
RETRY_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


class GroqError(RuntimeError):
    """Raised when Groq could not return a usable completion."""


def get_api_key():
    """Resolve the Groq key from the environment or Streamlit secrets.

    Streamlit Community Cloud supplies secrets through ``st.secrets`` rather
    than a ``.env`` file, so both sources are checked. The key is read on each
    call so a deployed app picks up a rotated secret without a code change.
    """
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key

    try:
        import streamlit as st

        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        # Outside a Streamlit runtime (CLI, evaluator, Kaggle) there are no
        # secrets to read, which is not an error.
        return None


def is_configured():
    """Return whether a live Groq call is possible."""
    return bool(get_api_key()) and requests is not None


def _extract_prompt_text(payload):
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    chunks = []

    for message in messages:
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, str):
            chunks.append(content)
            continue

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        chunks.append(text)

    return "\n".join(chunks)


def _build_offline_response(payload, reason=None):
    prompt_text = _extract_prompt_text(payload)

    diagnostic = f"\nGroq diagnostic: {reason}" if reason else ""

    context_match = re.search(
        r"Context:\s*(.*?)(?:\nQuestion:\s*|\Z)",
        prompt_text,
        flags=re.DOTALL,
    )
    question_match = re.search(r"Question:\s*(.*)", prompt_text, flags=re.DOTALL)

    context_text = context_match.group(1).strip() if context_match else ""
    question_text = question_match.group(1).strip() if question_match else ""

    if context_text:
        lines = [line.strip() for line in context_text.splitlines() if line.strip()]
        points = []
        for line in lines:
            cleaned = re.sub(r"\s+", " ", line)
            if cleaned and cleaned not in points:
                points.append(cleaned)
            if len(points) == 5:
                break

        if points:
            bullet_list = "\n".join(f"- {point}" for point in points)
            if question_text:
                return (
                    "Offline fallback response based on the retrieved document context.\n"
                    f"Question: {question_text}\n"
                    "Relevant points:\n"
                    f"{bullet_list}"
                    f"{diagnostic}"
                )

            return (
                "Offline fallback response based on the retrieved document context.\n"
                "Relevant points:\n"
                f"{bullet_list}"
                f"{diagnostic}"
            )

    if question_text:
        return (
            "Offline fallback response. "
            f"{reason or 'No GROQ_API_KEY is configured, so the app cannot call the hosted model.'}\n"
            f"Question: {question_text}"
        )

    return (
        "Offline fallback response. "
        f"{reason or 'No GROQ_API_KEY is configured, and no prompt context was available.'}"
    )


def _retry_delay(response, attempt):
    """Return how long to wait before retrying, preferring Groq's own advice."""
    retry_after = response.headers.get("retry-after") if response is not None else None
    if retry_after:
        try:
            return min(float(retry_after), MAX_BACKOFF_SECONDS)
        except ValueError:
            pass

    # Exponential backoff with jitter so concurrent viewers do not retry in lockstep.
    return min(2.0 ** attempt + random.uniform(0, 1), MAX_BACKOFF_SECONDS)


def query_groq_strict(payload, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """Call Groq, retrying transient failures, and raise if no answer arrives.

    Callers that index the result must use this rather than ``query_groq``: a
    fallback string would otherwise be embedded and stored as though it were a
    real summary, quietly corrupting retrieval.
    """
    api_key = get_api_key()
    if not api_key:
        raise GroqError("No GROQ_API_KEY is configured")
    if requests is None:
        raise GroqError("The requests package is unavailable")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_reason = "Groq request failed"
    attempts_made = 0
    for attempt in range(max_attempts):
        attempts_made = attempt + 1
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        except requests.RequestException as error:
            last_reason = f"Groq request failed ({type(error).__name__})"
            if attempt == max_attempts - 1:
                break
            time.sleep(_retry_delay(None, attempt))
            continue

        if response.status_code == 200:
            try:
                return response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise GroqError("Groq returned an unexpected response format") from error

        last_reason = f"Groq API returned HTTP {response.status_code}"
        if response.status_code not in RETRY_STATUS_CODES or attempt == max_attempts - 1:
            break
        time.sleep(_retry_delay(response, attempt))

    raise GroqError(f"{last_reason} after {attempts_made} attempt(s)")


def query_groq(payload):
    """Call Groq, degrading to a context-grounded offline answer on failure."""
    try:
        return query_groq_strict(payload)
    except GroqError as error:
        return _build_offline_response(payload, str(error))
