"""
Helpers for reading resume files and extracting textual content.

Resumes are typically uploaded in PDF format. This module contains a
lightweight parser built on top of PyPDF2 to extract text from PDF
documents. The text extraction is simplistic; for more robust parsing
consider integrating with an NLP library or specialised resume parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import logging

try:
    import PyPDF2
except Exception as e:
    logging.warning("PyPDF2 is not installed: %s", e)
    PyPDF2 = None


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract text from a PDF file using PyPDF2.

    Parameters
    ----------
    pdf_path: Path
        The path to the PDF file.

    Returns
    -------
    Optional[str]
        The concatenated textual content of the PDF, or None if extraction
        fails.
    """
    if PyPDF2 is None:
        return None
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                except Exception:
                    page_text = None
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        logging.warning("Failed to extract text from PDF %s: %s", pdf_path, e)
        return None
