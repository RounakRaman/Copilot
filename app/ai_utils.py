"""
Helper functions for running inference with local language models.

This module uses the Hugging Face ``transformers`` library to load
pretrained models for summarisation and answer generation. It defines a
single entry point, ``generate_answer()``, which orchestrates summarising
long inputs and crafting a prompt for a FLAN‑T5 model. The summariser
reduces long resumes and job descriptions into more digestible chunks.

Note: The models loaded here are relatively lightweight compared to
production‑grade LLMs. They run on CPU and have a limited context window.
If you have access to more powerful APIs (e.g. Google Gemini or
OpenAI's GPT‑4o) you can replace the generation call accordingly.
"""

from __future__ import annotations

import logging
from typing import Optional

from transformers import pipeline

# Create global pipelines on module import. Loading models can be slow, so
# performing it once avoids repeated overhead. Choose modest models that
# balance quality with inference speed on CPU.
try:
    summariser = pipeline("text-generation", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
except Exception as e:
    logging.warning("Failed to load summariser: %s", e)
    summariser = None

try:
    generator = pipeline("text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
except Exception as e:
    logging.warning("Failed to load generator: %s", e)
    generator = None


def summarise(text: str, max_length: int = 150) -> str:
    """
    Summarise a long piece of text into a shorter form.

    Uses the ``summariser`` pipeline if available; otherwise returns the
    original text. The ``max_length`` parameter controls the target length of
    the summary in tokens (approximate).
    """
    if summariser is None or not text:
        return text or ""
    # The huggingface summariser splits input longer than 1024 tokens across
    # multiple calls internally. Keep ``min_length`` relatively small to
    # encourage concise summaries.
    try:
        summary = summariser(text, max_length=max_length, min_length=max_length // 2, do_sample=False)
        if summary and isinstance(summary, list):
            return summary[0]["summary_text"]
    except Exception as e:
        logging.warning("Summarisation failed: %s", e)
    return text


def generate_answer(question: str,
                    resume_text: Optional[str] = None,
                    job_description: Optional[str] = None,
                    max_tokens: int = 200) -> str:
    """
    Generate a concise answer to a free‑form application question.

    Parameters
    ----------
    question: str
        The question to answer (e.g. "Why do you want to work here?").
    resume_text: Optional[str]
        Raw text extracted from the candidate's resume. May be None if no
        resume is provided.
    job_description: Optional[str]
        Raw text of the job description. May be None if unavailable.
    max_tokens: int
        Approximate upper bound on the number of tokens to generate.

    Returns
    -------
    str
        A generated answer tailored to the question and context.
    """
    context_parts = []
    if resume_text:
        context_parts.append(f"Resume Summary: {summarise(resume_text, max_length=200)}")
    if job_description:
        context_parts.append(f"Job Description Summary: {summarise(job_description, max_length=200)}")

    # Compose the prompt. Keeping the prompt explicit helps smaller models
    # understand what to do. The model will fill in the answer after the
    # "Answer:" cue.
    prompt_parts = [
        "You are assisting with a job application.",
        "Craft a professional, personalised answer to the question below.",
    ]
    if context_parts:
        prompt_parts.append("Use the following context:")
        prompt_parts.extend(context_parts)
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("Answer:")
    prompt = "\n".join(prompt_parts)

    # If no generator is available, return a fallback string instructing the
    # caller to supply their own AI key.
    if generator is None:
        return (
            "[AI unavailable] The local language model could not be loaded. "
            "Please install compatible transformers models or integrate your "
            "own AI API (e.g. Google Gemini) to enable answer generation."
        )
    try:
        output = generator(prompt, max_new_tokens=max_tokens, temperature=0.7)
        if output and isinstance(output, list):
            # Each result is a dict with a 'generated_text' key containing
            # the full prompt followed by the generated answer. Strip the
            # original prompt to return only the answer.
            generated = output[0]["generated_text"]
            # Remove the prompt prefix if present
            if generated.startswith(prompt):
                answer = generated[len(prompt):].strip()
            else:
                answer = generated.strip()
            return answer
    except Exception as e:
        logging.warning("Generation failed: %s", e)
    return "[AI unavailable] Failed to generate an answer."
