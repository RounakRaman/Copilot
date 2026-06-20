"""
Helper functions for generating job-application answers via a cloud LLM API.

This module previously loaded local Hugging Face models (bart-large-cnn +
flan-t5-base) at import time. That added ~2.5GB of RAM/disk and a long
model-download step, which is why deployment on Render's free tier
(512MB RAM) timed out on the port scan -- uvicorn never got a chance to
bind to $PORT because the import itself was still downloading/loading
weights, or it was OOM-killed.

This version calls Google's Gemini API instead. No local model weights,
no torch/transformers dependency, instant cold start. Gemini's free tier
(via Google AI Studio) does not require a credit card.

Setup:
    1. Get a free API key: https://aistudio.google.com/app/apikey
    2. Set it as an environment variable on Render (Dashboard -> your
       service -> Environment -> Add Environment Variable):
           GEMINI_API_KEY = <your key>
    3. That's it. No other config needed.

If GEMINI_API_KEY is not set, generate_answer() and summarise() return a
clear fallback string instead of crashing, so the rest of the app
(profile storage, form filling for non-AI fields) keeps working.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Use the Flash model: fast, cheap/free, generous rate limits. Swap to
# "gemini-1.5-pro-latest" if you want higher quality at lower throughput.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

REQUEST_TIMEOUT_SECONDS = 30


def _call_gemini(prompt: str, max_output_tokens: int = 300, temperature: float = 0.7) -> Optional[str]:
    """
    Low-level helper: send a single prompt to Gemini and return the text
    response, or None if the call fails for any reason (missing key,
    network error, bad response shape, etc.). Never raises.
    """
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY is not set; cannot call Gemini.")
        return None

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    try:
        response = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            logging.warning("Gemini returned no candidates: %s", data)
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except requests.exceptions.Timeout:
        logging.warning("Gemini request timed out after %ss", REQUEST_TIMEOUT_SECONDS)
        return None
    except requests.exceptions.RequestException as e:
        logging.warning("Gemini request failed: %s", e)
        return None
    except (KeyError, ValueError, IndexError) as e:
        logging.warning("Unexpected Gemini response shape: %s", e)
        return None


def summarise(text: str, max_length: int = 150) -> str:
    """
    Summarise a long piece of text into a shorter form using Gemini.

    Falls back to returning the original text (untouched) if the API
    call fails or no key is configured -- summarisation is a nice-to-have,
    not a hard requirement, so we never block the caller on it.

    ``max_length`` is treated as an approximate target word count, since
    Gemini doesn't take a token cap for this kind of instruction directly.
    """
    if not text:
        return ""
    prompt = (
        f"Summarise the following text in no more than {max_length} words. "
        f"Keep only the most relevant facts for a job application context.\n\n"
        f"Text:\n{text}"
    )
    summary = _call_gemini(prompt, max_output_tokens=max_length * 2)
    return summary if summary else text


def generate_answer(question: str,
                    resume_text: Optional[str] = None,
                    job_description: Optional[str] = None,
                    max_tokens: int = 200) -> str:
    """
    Generate a concise, tailored answer to a free-form application question.

    Parameters
    ----------
    question: str
        The question to answer (e.g. "Why do you want to work here?").
    resume_text: Optional[str]
        Raw text extracted from the candidate's resume. May be None.
    job_description: Optional[str]
        Raw text of the job description. May be None.
    max_tokens: int
        Approximate upper bound on the length of the generated answer.

    Returns
    -------
    str
        A generated answer, or a clearly-labelled fallback string if
        generation isn't possible (no API key, network failure, etc.).
        The fallback is intentional: this is a human-in-the-loop tool,
        so a visible placeholder is safer than silently failing or
        submitting blank text.
    """
    context_parts = []
    if resume_text:
        context_parts.append(f"Resume summary: {summarise(resume_text, max_length=200)}")
    if job_description:
        context_parts.append(f"Job description summary: {summarise(job_description, max_length=200)}")

    prompt_parts = [
        "You are helping a candidate answer a job application question.",
        "Write a professional, specific, first-person answer. Avoid generic "
        "filler -- ground the answer in the resume and job description "
        "context given below where relevant.",
    ]
    if context_parts:
        prompt_parts.append("Context:")
        prompt_parts.extend(context_parts)
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("Answer:")
    prompt = "\n".join(prompt_parts)

    answer = _call_gemini(prompt, max_output_tokens=max_tokens)
    if answer:
        return answer

    return (
        "[AI unavailable] Could not generate an answer. Check that "
        "GEMINI_API_KEY is set correctly in your environment, and that "
        "you have not exceeded the free-tier rate limit."
    )
