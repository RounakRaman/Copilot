"""
Utilities for extracting text from images using OCR.

This module wraps the EasyOCR library to provide simple functions for
converting screenshots or other image data into plain text. OCR is
useful when job descriptions or company information are embedded as
images in application forms.
"""

from __future__ import annotations

import io
import logging
from typing import List

try:
    import easyocr  # type: ignore
    _reader = easyocr.Reader(["en"])
except Exception as e:
    logging.warning("Failed to initialise EasyOCR: %s", e)
    easyocr = None  # type: ignore
    _reader = None


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    Run OCR on an in‑memory image and return concatenated text.

    Parameters
    ----------
    image_bytes: bytes
        Raw image data in any format supported by EasyOCR (JPEG, PNG, etc.).

    Returns
    -------
    str
        The detected text, or an empty string if OCR is unavailable.
    """
    if _reader is None:
        return ""
    try:
        results: List[str] = []
        # EasyOCR expects a file path or an ndarray; we can pass bytes
        # directly and it will detect the format automatically.
        for line in _reader.readtext(image_bytes, detail=0):
            results.append(line)
        return "\n".join(results)
    except Exception as e:
        logging.warning("OCR extraction failed: %s", e)
        return ""
