"""
Module for automating job application form filling using Playwright.

The core entry point is ``fill_form()``, which accepts a URL, a profile
dictionary, an optional path to a resume file and an optional job
description. It launches a headless browser, navigates to the form and
attempts to populate fields based on their labels or placeholders. The
function returns a report of what was filled. It deliberately avoids
submitting the form to keep the human in the loop.

This module depends on Playwright, which must be installed separately via
``pip install playwright`` and initialised using ``playwright install``.
The automation is best‑effort; complex or highly customised forms may
require additional logic. Feel free to extend this module with more
intelligent field matching, error handling and support for other form
frameworks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Optional

from . import ai_utils

# We import Playwright lazily so that the module can be imported without
# Playwright installed. When the fill_form function is invoked the import
# will occur; if Playwright is unavailable a descriptive error is raised.

async def _fill_form_async(url: str,
                           profile: Dict[str, str],
                           resume_path: Optional[Path],
                           job_description: Optional[str]) -> Dict[str, str]:
    """
    Internal coroutine that performs the browser automation.
    See ``fill_form`` for parameter descriptions.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Playwright is not installed. Please run `pip install playwright` "
            "and `playwright install` to enable form automation."
        )

    # Dict to return with field names and values that were filled
    filled: Dict[str, str] = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)

        # Wait for the form to be visible. We assume there is at least one
        # <input> element on the page. Adjust timeout as needed.
        await page.wait_for_selector("input, textarea", timeout=15000)

        # Identify all input and textarea elements. We'll iterate through them
        # and fill based on placeholder/labels matching profile keys.
        inputs = page.locator("input")
        textareas = page.locator("textarea")

        # Helper to normalise a string for matching (lowercase, strip).
        def norm(s: str) -> str:
            return s.lower().strip() if s else ""

        # Build a mapping of possible keywords to profile fields for quick lookup
        keyword_map = {
            "name": profile.get("name"),
            "full name": profile.get("name"),
            "first name": profile.get("name"),
            "last name": profile.get("name"),
            "email": profile.get("email"),
            "email address": profile.get("email"),
            "phone": profile.get("phone"),
            "phone number": profile.get("phone"),
            "contact number": profile.get("phone"),
            "linkedin": profile.get("linkedin"),
            "linkedin profile": profile.get("linkedin"),
            "website": profile.get("portfolio"),
            "portfolio": profile.get("portfolio"),
            "github": profile.get("github"),
            "current company": profile.get("current_company"),
            "current employer": profile.get("current_company"),
            "current role": profile.get("current_role"),
            "job title": profile.get("current_role"),
            "experience": profile.get("experience_years"),
        }

        # Fill simple input fields
        count = await inputs.count()
        for i in range(count):
            inp = inputs.nth(i)
            # Skip hidden or disabled inputs
            try:
                input_type = await inp.get_attribute("type")
                disabled = await inp.get_attribute("disabled")
                hidden = await inp.get_attribute("hidden")
            except Exception:
                continue
            if hidden or disabled:
                continue

            # File upload
            if input_type and input_type.lower() == "file" and resume_path:
                await inp.set_input_files(str(resume_path))
                filled["resume"] = str(resume_path)
                continue

            # Determine label or placeholder for the input
            placeholder = await inp.get_attribute("placeholder") or ""
            # Also attempt to find an associated <label> by id
            label_text = ""
            id_attr = await inp.get_attribute("id")
            if id_attr:
                # Use CSS to find <label for="id">
                label_locator = page.locator(f"label[for='{id_attr}']")
                if await label_locator.count() > 0:
                    try:
                        label_text = await label_locator.nth(0).inner_text()
                    except Exception:
                        label_text = ""

            field_name = norm(placeholder) or norm(label_text)
            if not field_name:
                continue

            # Attempt to match field name with profile keywords
            for keyword, value in keyword_map.items():
                if value and keyword in field_name:
                    await inp.fill(value)
                    filled[keyword] = value
                    break

        # Fill textareas (often for open‑ended questions)
        tcount = await textareas.count()
        for i in range(tcount):
            area = textareas.nth(i)
            placeholder = await area.get_attribute("placeholder") or ""
            label_text = ""
            # Try to find parent label
            parent_label = await area.evaluate("el => el.previousElementSibling && el.previousElementSibling.tagName === 'LABEL' ? el.previousElementSibling.innerText : ''")
            label_text = parent_label or placeholder
            field_name = norm(label_text)
            if not field_name:
                continue

            # Determine if this is a typical question requiring an AI answer
            # Example keywords: why, describe, tell us, motivation
            if any(k in field_name for k in ["why", "describe", "tell", "motivation", "introduce", "about you"]):
                answer = ai_utils.generate_answer(
                    question=label_text or "",
                    resume_text=profile.get("resume_text"),
                    job_description=job_description,
                    max_tokens=200,
                )
                await area.fill(answer)
                filled[field_name] = answer
            # Map to known profile fields if not open ended
            else:
                for keyword, value in keyword_map.items():
                    if value and keyword in field_name:
                        await area.fill(value)
                        filled[keyword] = value
                        break

        # Do not submit the form; we return the filled fields for review
        await browser.close()
        return filled


def fill_form(url: str,
              profile: Dict[str, str],
              resume_path: Optional[Path] = None,
              job_description: Optional[str] = None) -> Dict[str, str]:
    """
    Public API for form filling.

    Parameters
    ----------
    url: str
        The URL of the job application form to open.
    profile: Dict[str, str]
        The candidate's profile fields, typically obtained via the profile API.
    resume_path: Optional[Path]
        Path to the resume file to upload if the form supports it.
    job_description: Optional[str]
        Text of the job description, used to generate answers for open‑ended
        questions.

    Returns
    -------
    Dict[str, str]
        A mapping of field identifiers to the values that were inserted.
    """
    # Run the async coroutine in an event loop. If an event loop is already
    # running (e.g. within Jupyter), create a new one to avoid conflicts.
    try:
        return asyncio.run(_fill_form_async(url, profile, resume_path, job_description))
    except RuntimeError:
        # Handle case where event loop is already running
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_fill_form_async(url, profile, resume_path, job_description))
        loop.close()
        return result
