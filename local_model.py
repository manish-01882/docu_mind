"""Local Gemma inference for Smart Doc and Kaggle notebooks.

Set ``KAGGLE_MODEL_PATH`` (or ``LOCAL_MODEL_PATH``) to an attached Gemma 3
Transformers model directory. When neither is set, the Hugging Face model ID in
``LOCAL_MODEL_ID`` is used, which may require an authenticated download.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_MODEL_ID = "google/gemma-3-4b-it"
DEFAULT_MAX_CONTEXT_CHARS = 16_000
DEFAULT_MAX_CONTEXT_IMAGES = 2


class LocalModelError(RuntimeError):
    """Raised when the configured local model cannot be loaded or queried."""


def _env_positive_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise LocalModelError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise LocalModelError(f"{name} must be a positive integer")
    return parsed


def get_model_reference() -> str:
    """Return the local model directory or model ID selected by the environment."""
    return (
        os.getenv("KAGGLE_MODEL_PATH")
        or os.getenv("LOCAL_MODEL_PATH")
        or os.getenv("LOCAL_MODEL_ID")
        or DEFAULT_MODEL_ID
    )


def _dtype_for_device(torch):
    if not torch.cuda.is_available():
        return torch.float32
    # Kaggle GPUs differ. Prefer bf16 when supported; fp16 keeps the 4B model
    # usable on older P100/T4 sessions.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@lru_cache(maxsize=1)
def load_local_model():
    """Load the instruction-tuned Gemma model once per Python process."""
    try:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    except ImportError as error:
        raise LocalModelError(
            "Local inference requires torch, transformers>=4.50.0, and accelerate. "
            "Install requirements.txt before running the app or evaluator."
        ) from error

    model_reference = get_model_reference()
    model_path = Path(model_reference)
    if (os.getenv("KAGGLE_MODEL_PATH") or os.getenv("LOCAL_MODEL_PATH")) and not model_path.is_dir():
        raise LocalModelError(
            f"Configured local model directory does not exist: {model_reference}. "
            "Attach Gemma 3 in Kaggle and set KAGGLE_MODEL_PATH to the returned directory."
        )

    dtype = _dtype_for_device(torch)
    try:
        processor = AutoProcessor.from_pretrained(model_reference)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_reference,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        ).eval()
    except Exception as error:
        raise LocalModelError(
            f"Could not load local model '{model_reference}'. In Kaggle, request Gemma 3 access, "
            "attach google/gemma-3/transformers/gemma-3-4b-it, and set KAGGLE_MODEL_PATH."
        ) from error

    return model, processor, dtype


def _model_device(model):
    return next(model.parameters()).device


def _decode_image(image_base64: str) -> Image.Image:
    payload = image_base64.split(",", 1)[-1] if "," in image_base64 else image_base64
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


def _generate(messages, *, max_new_tokens: int) -> str:
    import torch

    model, processor, dtype = load_local_model()
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = _model_device(model)
        inputs = inputs.to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        return processor.decode(generated[0][input_length:], skip_special_tokens=True).strip()
    except Exception as error:
        raise LocalModelError("Local Gemma generation failed") from error


def summarize_text(text: str) -> str:
    """Summarize a text or HTML table chunk for semantic retrieval."""
    if not isinstance(text, str) or not text.strip():
        return ""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Summarize the following document chunk concisely for semantic retrieval. "
                        "Preserve key facts, figures, and relationships. Return only the summary.\n\n"
                        f"Document chunk:\n{text}"
                    ),
                }
            ],
        }
    ]
    return _generate(messages, max_new_tokens=192)


def summarize_image(image_base64: str) -> str:
    """Describe an extracted image so it can be retrieved through its summary."""
    image = _decode_image(image_base64)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Describe this document image concisely for semantic retrieval. "
                        "Include visible labels, values, trends, and relationships when present."
                    ),
                },
            ],
        }
    ]
    return _generate(messages, max_new_tokens=160)


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
    """Answer only from retrieved text/image context using local Gemma inference."""
    context = _context_text(texts)
    prompt = (
        "Answer the question using only the supplied source context. "
        "If the sources do not contain the answer, say that the context is insufficient. "
        "Do not invent facts.\n\n"
        f"Sources:\n{context or '[No text source was retrieved.]'}\n\n"
        f"Question: {question}"
    )
    content = [{"type": "text", "text": prompt}]
    image_limit = _env_positive_int("LOCAL_MAX_CONTEXT_IMAGES", DEFAULT_MAX_CONTEXT_IMAGES)
    for image_base64 in list(images)[:image_limit]:
        content.append({"type": "image", "image": _decode_image(image_base64)})

    return _generate([{"role": "user", "content": content}], max_new_tokens=320)


def answer_chat(history: Iterable[tuple[str, str]]) -> str:
    """Generate a direct local chat response from the app's tuple history."""
    messages = []
    for role, text in history:
        if role not in {"user", "assistant"}:
            raise LocalModelError(f"Unsupported chat role: {role}")
        messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    if not messages:
        raise LocalModelError("Cannot answer an empty chat history")
    return _generate(messages, max_new_tokens=320)
