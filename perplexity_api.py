"""Perplexity API wrapper with an offline fallback."""

from __future__ import annotations

import os
import re

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
API_KEY = os.getenv("PERPLEXITY_API_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"


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


def _build_offline_response(payload):
    prompt_text = _extract_prompt_text(payload)

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
                )

            return (
                "Offline fallback response based on the retrieved document context.\n"
                "Relevant points:\n"
                f"{bullet_list}"
            )

    if question_text:
        return (
            "Offline fallback response. No Perplexity API key is configured, so the app "
            "cannot call the hosted model.\n"
            f"Question: {question_text}"
        )

    return (
        "Offline fallback response. No Perplexity API key is configured, and no prompt "
        "context was available."
    )


def query_perplexity(payload):
    if not API_KEY or requests is None:
        return _build_offline_response(payload)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException:
        pass

    return _build_offline_response(payload)
