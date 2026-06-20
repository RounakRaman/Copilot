"""
Utilities for extracting text from images using OCR.

This module previously used EasyOCR, which depends on PyTorch under the
hood even though "torch" was never listed directly in requirements.txt.
That's the same failure mode as the old ai_utils.py: a large model
(~100MB+ of weights, plus the torch runtime itself) loaded at import
time, which either blows past Render's free-tier 512MB RAM limit or
takes long enough that the port-scan timeout kills the deploy before
uvicorn ever binds to $PORT.

This version calls the OCR.space API instead -- a free-tier OCR service
that requires no local model weights and no torch dependency. Import is
instant; there is nothing to load.

Setup:
    1. Get a free API key (no credit card required):
       https://ocr.space/ocrapi/freekey
    2. Set it as an environment variable on Render (Dashboard -> your
       service -> Environment -> Add Environment Variable):
           OCR_SPACE_API_KEY = <your key>
    3. That's it.

Free tier limits (as of writing): 25,000 requests/month, 1MB max file
size, 3 requests/second. Fine for a personal job-application tool;
revisit if you ever scale this beyond yourself.

If OCR_SPACE_API_KEY is not set, extract_text_from_image_bytes() returns
an empty string instead of crashing, so the rest of the app keeps
working -- OCR is only used as a fallback when a job description is
embedded as an image.
"""

from __future__ import annotations

import base64
import logging
import os

import requests

OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")
OCR_SPACE_URL = "https://api.ocr.space/parse/image"

REQUEST_TIMEOUT_SECONDS = 30

# OCR.space free tier caps uploads around 1MB. Guard against silently
# sending an oversized request that just fails.
MAX_IMAGE_BYTES = 1_000_000


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    Run OCR on an in-memory image and return concatenated text.

    Parameters
    ----------
    image_bytes: bytes
        Raw image data (JPEG, PNG, etc.).

    Returns
    -------
    str
        The detected text, or an empty string if OCR is unavailable,
        misconfigured, or the request fails for any reason. This function
        is intentionally non-raising -- a missing job-description image
        should never crash the autofill flow.
    """
    if not OCR_SPACE_API_KEY:
        logging.warning("OCR_SPACE_API_KEY is not set; skipping OCR.")
        return ""

    if not image_bytes:
        return ""

    if len(image_bytes) > MAX_IMAGE_BYTES:
        logging.warning(
            "Image is %d bytes, over the %d byte free-tier limit; skipping OCR.",
            len(image_bytes), MAX_IMAGE_BYTES,
        )
        return ""

    try:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "base64Image": f"data:image/png;base64,{encoded}",
            "OCREngine": "2",
        }
        response = requests.post(OCR_SPACE_URL, data=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        if data.get("IsErroredOnProcessing"):
            logging.warning("OCR.space reported an error: %s", data.get("ErrorMessage"))
            return ""

        results = data.get("ParsedResults") or []
        texts = [r.get("ParsedText", "") for r in results]
        return "\n".join(t.strip() for t in texts if t.strip())
    except requests.exceptions.Timeout:
        logging.warning("OCR.space request timed out after %ss", REQUEST_TIMEOUT_SECONDS)
        return ""
    except requests.exceptions.RequestException as e:
        logging.warning("OCR.space request failed: %s", e)
        return ""
    except (KeyError, ValueError) as e:
        logging.warning("Unexpected OCR.space response shape: %s", e)
        return ""
