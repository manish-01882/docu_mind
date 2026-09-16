"""Model router for summarisation and answering.

This module mirrors the public API of ``local_model`` so every call site can
stay backend-agnostic. Deployments call Groq, which needs no GPU and keeps the
container small enough for a free CPU tier. The local Gemma path is retained
because the Kaggle benchmark in ``evaluation/KAGGLE.md`` runs without an API
key; select it with ``SUMMARIZER_BACKEND=local`` or by setting a model path.
"""

from __future__ import annotations

import os
from typing import Iterable

from groq_api import TEXT_MODEL, VISION_MODEL, GroqError, query_groq, query_groq_strict


DEFAULT_MAX_CONTEXT_CHARS = 16_000
DEFAULT_MAX_CONTEXT_IMAGES = 2
# Groq rejects requests carrying more than five images.
GROQ_MAX_IMAGES = 5

SUMMARIZE_TEXT_PROMPT = (
    "Summarize the following document chunk concisely for semantic retrieval. "
    "Preserve key facts, figures, and relationships. Return only the summary.\n\n"
    "Document chunk:\n"
)

SUMMARIZE_IMAGE_PROMPT = (
    "Describe this document image concisely for semantic retrieval. "
    "Include visible labels, values, trends, and relationships when present."
)

ANSWER_PROMPT = (
    "Answer the question using only the supplied source context. "
    "If the sources do not contain the answer, say that the context is insufficient. "
    "Do not invent facts."
)


class InferenceError(RuntimeError):
    """Raised when an element could not be summarised or answered."""


def use_local_backend() -> bool:
    """Return whether inference should run through local Gemma weights."""
    backend = os.getenv("SUMMARIZER_BACKEND", "").strip().lower()
    if backend == "local":
        return True
    if backend == "groq":
        return False
    # With no explicit choice, use local weights only when they are configured,
    # which is how the Kaggle notebook runs.
    return bool(os.getenv("KAGGLE_MODEL_PATH") or os.getenv("LOCAL_MODEL_PATH"))


def _env_positive_int(name: str, default: int) -> int:
    try:
        parsed = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise InferenceError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise InferenceError(f"{name} must be a positive integer")
    return parsed


def _strip_data_uri(image_base64: str) -> str:
    return image_base64.split(",", 1)[-1] if "," in image_base64 else image_base64


def _image_part(image_base64: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{_strip_data_uri(image_base64)}"},
    }


def _ask_groq(content: list[dict], *, has_images: bool) -> str:
    return query_groq(
        {
            "model": VISION_MODEL if has_images else TEXT_MODEL,
            "messages": [{"role": "user", "content": content}],
        }
    )


def _summarize_with_groq(content: list[dict], *, has_images: bool) -> str:
    """Summarise strictly: a failure must not be indexed as if it were a summary."""
    payload = {
        "model": VISION_MODEL if has_images else TEXT_MODEL,
        "messages": [{"role": "user", "content": content}],
    }
    try:
        return query_groq_strict(payload)
    except GroqError as error:
        raise InferenceError(str(error)) from error


def summarize_text(text: str) -> str:
    """Summarise a text chunk or an HTML table for the retrieval index."""
    if not isinstance(text, str) or not text.strip():
        return ""

    if use_local_backend():
        from local_model import summarize_text as _local

        return _local(text)

    return _summarize_with_groq(
        [{"type": "text", "text": f"{SUMMARIZE_TEXT_PROMPT}{text}"}],
        has_images=False,
    )


def summarize_image(image_base64: str) -> str:
    """Describe an extracted image so it can be retrieved through its summary."""
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise InferenceError("Image payload was empty")

    if use_local_backend():
        from local_model import LocalModelError
        from local_model import summarize_image as _local

        try:
            return _local(image_base64)
        except LocalModelError as error:
            raise InferenceError(str(error)) from error

    return _summarize_with_groq(
        [
            {"type": "text", "text": SUMMARIZE_IMAGE_PROMPT},
            _image_part(image_base64),
        ],
        has_images=True,
    )


def _context_text(texts: Iterable[object]) -> str:
    budget = _env_positive_int("LOCAL_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS)
    blocks = []
    used = 0
    for index, text in enumerate(texts, start=1):
        content = getattr(text, "page_content", str(text)).strip()
        if not content:
            continue
        remaining = budget - used
        if remaining <= 0:
            break
        block = f"[Source {index}]\n{content[:remaining]}"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def answer_question(question: str, texts: Iterable[object] = (), images: Iterable[str] = ()) -> str:
    """Answer only from retrieved text and image context."""
    if use_local_backend():
        from local_model import answer_question as _local

        return _local(question, texts=texts, images=images)

    context = _context_text(texts)
    prompt = (
        f"{ANSWER_PROMPT}\n\n"
        f"Sources:\n{context or '[No text source was retrieved.]'}\n\n"
        f"Question: {question}"
    )
    content = [{"type": "text", "text": prompt}]

    image_limit = min(
        _env_positive_int("LOCAL_MAX_CONTEXT_IMAGES", DEFAULT_MAX_CONTEXT_IMAGES),
        GROQ_MAX_IMAGES,
    )
    selected = [image for image in list(images)[:image_limit] if isinstance(image, str) and image.strip()]
    content.extend(_image_part(image) for image in selected)

    return _ask_groq(content, has_images=bool(selected))


def answer_chat(history: Iterable[tuple[str, str]]) -> str:
    """Generate a direct chat response from the app's tuple history."""
    if use_local_backend():
        from local_model import answer_chat as _local

        return _local(history)

    messages = []
    for role, text in history:
        if role not in {"user", "assistant"}:
            raise InferenceError(f"Unsupported chat role: {role}")
        messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    if not messages:
        raise InferenceError("Cannot answer an empty chat history")

    return query_groq({"model": TEXT_MODEL, "messages": messages})
