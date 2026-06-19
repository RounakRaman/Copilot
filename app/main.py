"""
Main entry point for the AI Job Application Copilot service.

This module spins up a FastAPI server exposing a handful of REST endpoints
that wrap the functionality needed for an automated job application flow.
Endpoints include:

* `/profile` – return the persisted static profile information or accept
  updates to it. Profiles are stored locally in a JSON file so that you
  only have to enter your name, email, phone number and other details once.
* `/generate-answer` – accept a question and optionally a path to a resume
  and job description; return a tailored answer using the local LLM. This
  endpoint orchestrates the summarisation and generation logic from
  ``ai_utils``.
* `/autofill` – given a URL to a job application form, invoke the
  Playwright automation in ``form_filler`` to detect fields, upload the
  appropriate resume and populate all repetitive fields. This endpoint does
  not submit the application automatically; instead it prepares a draft and
  returns control to the caller so the user can review before submitting.

The API is deliberately minimal and opinionated. It is intended as a
starting point for a more fully‑featured agentic system, not a final
product. Feel free to extend it for your own needs.
"""

import json
import os
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from . import ai_utils, resume_utils, ocr_utils, form_filler

# The location of the profile JSON file. Modify this if you want to
# persist profiles elsewhere.
PROFILE_PATH = Path(__file__).resolve().parent / "profile.json"


class Profile(BaseModel):
    """Pydantic model representing the user's static information."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    portfolio: Optional[str] = None
    github: Optional[str] = None
    current_company: Optional[str] = None
    current_role: Optional[str] = None
    experience_years: Optional[str] = None
    notice_period: Optional[str] = None
    current_ctc: Optional[str] = None
    expected_ctc: Optional[str] = None
    work_authorization: Optional[str] = None
    education: Optional[List[str]] = None
    skills: Optional[List[str]] = None


class GenerateAnswerRequest(BaseModel):
    """
    Request schema for generating a customised answer.

    * ``question`` – the free‑form question asked on the job application.
    * ``job_description`` – optional raw text description of the job; if
      omitted the model will fall back to using only the resume.
    * ``resume_path`` – optional path to a local resume PDF; if not
      provided the service uses the default resume stored in ``profile.json``.
    * ``max_tokens`` – optional token budget for the generation.
    """

    question: str
    job_description: Optional[str] = None
    resume_path: Optional[str] = None
    max_tokens: Optional[int] = 200


class GenerateAnswerResponse(BaseModel):
    """
    Response schema for the generate‑answer endpoint.

    * ``answer`` – the AI‑generated answer tailored to the question and
      context.
    """

    answer: str


app = FastAPI(title="AI Job Application Copilot",
              description=(
                  "A lightweight agentic service for automating job "
                  "applications. Stores your static profile, reads your "
                  "resume, parses job descriptions and generates tailored "
                  "answers using a local language model."
              ),
              version="0.1.0")


def load_profile() -> Profile:
    """Load the user profile from disk or return an empty profile."""
    if PROFILE_PATH.exists():
        data = json.loads(PROFILE_PATH.read_text())
        return Profile(**data)
    return Profile()


def save_profile(profile: Profile) -> None:
    """Persist the profile to disk."""
    PROFILE_PATH.write_text(profile.json(indent=2))


@app.get("/profile", response_model=Profile)
def get_profile() -> Profile:
    """Return the stored profile. Create an empty profile if none exists."""
    return load_profile()


@app.post("/profile", response_model=Profile)
def update_profile(profile: Profile) -> Profile:
    """
    Replace the existing profile with the provided one. This operation will
    overwrite all fields; to update a subset of fields read the current
    profile, modify it client‑side and submit the full object.
    """
    save_profile(profile)
    return profile


@app.post("/generate-answer", response_model=GenerateAnswerResponse)
async def generate_answer(request: GenerateAnswerRequest) -> GenerateAnswerResponse:
    """
    Generate a tailored answer to a free‑form question using the resume and
    job description.

    The service will attempt to read the resume file specified in
    ``request.resume_path``. If omitted, it falls back to the default
    ``resume_path`` stored in the profile. The job description is optional; if
    provided, it is summarised and included in the prompt. The underlying
    generation uses the lightweight FLAN‑T5 model bundled with this repo.
    """
    # Determine which resume to use
    profile = load_profile()
    resume_path: Optional[Path] = None
    if request.resume_path:
        resume_path = Path(request.resume_path)
    elif profile and profile.github:  # we use github field as placeholder for resume path
        # Overload: store resume path in github field for demonstration
        resume_path = Path(profile.github)

    resume_text = None
    if resume_path and resume_path.exists():
        resume_text = resume_utils.extract_text_from_pdf(resume_path)

    # Generate the answer
    answer = ai_utils.generate_answer(
        question=request.question,
        resume_text=resume_text,
        job_description=request.job_description,
        max_tokens=request.max_tokens or 200,
    )
    return GenerateAnswerResponse(answer=answer)


@app.post("/autofill")
async def autofill_job_application(url: str = Form(...),
                                   resume_path: Optional[str] = Form(None),
                                   screenshot: Optional[UploadFile] = File(None)):
    """
    Launch Playwright to open the provided application form URL and attempt to
    populate it using the saved profile and a selected resume. Optionally
    accepts a screenshot of the form if the text is not machine‑readable; the
    screenshot is run through OCR to extract the job description.

    The endpoint returns a dictionary summarising what was filled. It does not
    submit the form, leaving the final review and submission to the user.
    """
    profile = load_profile()

    # Determine job description if a screenshot is provided
    job_desc_text: Optional[str] = None
    if screenshot:
        image_bytes = await screenshot.read()
        job_desc_text = ocr_utils.extract_text_from_image_bytes(image_bytes)

    # Determine the resume to upload
    resume_file_path = None
    if resume_path:
        resume_file_path = Path(resume_path)
    elif profile.github:
        resume_file_path = Path(profile.github)

    # Invoke the automation. This will return a dictionary of filled fields.
    fill_report = form_filler.fill_form(
        url=url,
        profile=profile.dict(),
        resume_path=resume_file_path,
        job_description=job_desc_text,
    )
    return fill_report
