"""
vision.py — Image → text description using Llama 4 Scout on Groq.

Handles file paths, URLs, and raw bytes.
Caches descriptions by image hash to avoid redundant API calls.
"""
import urllib.request
import base64
import hashlib
import io
import logging
from pathlib import Path
from typing import Union

import requests
from PIL import Image
from groq import Groq

from config import GROQ_API_KEY, VISION_MODEL, VISION_MAX_TOKENS, VISION_TEMPERATURE

logger = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_API_KEY)

VISION_PROMPT = """
You are an expert e-commerce product analyst.
Analyze this product image and provide a detailed description optimized for search.

Include:
1. Product category (e.g., sneakers, laptop, headphones, clothing)
2. Brand (if visible, otherwise say 'Unknown brand')
3. Color(s) and material
4. Key visible features and design elements
5. Apparent use case (e.g., running, gaming, casual wear)
6. Price range estimate (budget/mid-range/premium)

Format: Write 3-4 sentences as a product description that would help find similar items.
Be specific and factual. Do not make up specs not visible in the image.
"""

_cache: dict = {}  # image_hash → description


def _load_image_bytes(source: Union[str, bytes, Path]) -> tuple:
    """
    Load image bytes from path, URL, or bytes.
    Converts ANY format to JPEG using a robust fallback chain.
    """
    if isinstance(source, bytes):
        raw_bytes  = source
        media_type = "image/jpeg"

    elif isinstance(source, Path) or (isinstance(source, str) and not source.startswith("http")):
        path = str(source)
        with open(path, "rb") as f:
            raw_bytes = f.read()
        ext = path.lower().rsplit(".", 1)[-1]
        media_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp",
        }.get(ext, "image/jpeg")

    else:
        # URL — use urllib with a browser User-Agent (some servers block requests)
        req = urllib.request.Request(
            source,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw_bytes  = resp.read()
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            media_type = content_type.split(";")[0].strip()

    # ── Convert anything not supported by Groq → JPEG ─────────────────────────
    # Groq vision accepts only: image/jpeg, image/png, image/webp, image/gif
    supported = {"image/jpeg", "image/png", "image/webp", "image/gif"}

    if media_type not in supported:
        try:
            # Try Pillow first
            img = Image.open(io.BytesIO(raw_bytes))
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            raw_bytes  = buf.getvalue()
            media_type = "image/jpeg"

        except Exception:
            try:
                # Pillow failed (e.g. AVIF) — try pillow-avif-plugin
                import pillow_avif  # registers AVIF handler into Pillow
                img = Image.open(io.BytesIO(raw_bytes))
                img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=92)
                raw_bytes  = buf.getvalue()
                media_type = "image/jpeg"

            except Exception:
                # Last resort — use opencv
                try:
                    import numpy as np
                    import cv2
                    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    _, buf = cv2.imencode(".jpg", img)
                    raw_bytes  = buf.tobytes()
                    media_type = "image/jpeg"
                except Exception as e:
                    raise ValueError(
                        f"Cannot decode image (tried Pillow, pillow-avif, opencv). "
                        f"Error: {e}. Try saving the image as JPG/PNG and uploading directly."
                    )

    return raw_bytes, media_type

def _resize_if_needed(image_bytes: bytes, max_side: int = 800) -> tuple:
    """Resize large images to stay within Groq's ~4MB base64 limit."""
    img = Image.open(io.BytesIO(image_bytes))
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue(), "image/jpeg"
    return image_bytes, None  # unchanged


def describe_product_image(
    source: Union[str, bytes, Path],
    use_cache: bool = True
) -> str:
    """
    Describe a product image using Llama 4 Scout vision on Groq.

    Args:
        source:    File path, URL, or raw bytes
        use_cache: Cache results by image hash (avoids duplicate API calls)

    Returns:
        Detailed product description string
    """
    image_bytes, media_type = _load_image_bytes(source)

    # Optional resize
    resized, new_mt = _resize_if_needed(image_bytes)
    if new_mt:
        image_bytes, media_type = resized, new_mt

    # Cache check
    if use_cache:
        cache_key = hashlib.md5(image_bytes).hexdigest()
        if cache_key in _cache:
            logger.debug("Returning cached image description")
            return _cache[cache_key]

    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    logger.info(f"Calling {VISION_MODEL} for image description...")
    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_str}"}},
                {"type": "text",      "text": VISION_PROMPT}
            ]
        }],
        max_tokens=VISION_MAX_TOKENS,
        temperature=VISION_TEMPERATURE
    )

    description = response.choices[0].message.content.strip()
    logger.info(f"Image described: {description[:80]}...")

    if use_cache:
        _cache[cache_key] = description

    return description
