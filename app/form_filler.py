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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
#
# NOTE: there used to be a synchronous `fill_form()` wrapper here that
# tried to manage its own asyncio event loop (asyncio.run(), with a
# fallback to asyncio.new_event_loop() if a loop was already running).
# That fallback was broken: loop.run_until_complete() ALSO requires that
# no loop is currently running on the thread, so calling this from within
# main.py's `async def autofill_job_application(...)` -- which already
# runs on uvicorn's event loop -- raised:
#     RuntimeError: Cannot run the event loop while another loop is running
#
# There is no way to "nest" a new event loop on top of a running one in
# plain asyncio. The actual fix is to not create a second loop at all:
# since the caller is already async, it should just `await` this
# coroutine directly. main.py has been updated accordingly to call
# `await form_filler._fill_form_async(...)` instead of a sync fill_form().
#
# _fill_form_async is intentionally the public entry point now. If you
# ever need a sync (non-async) caller for this module, use
# `asyncio.run(_fill_form_async(...))` from a plain sync context that is
# NOT already inside a running event loop -- not from inside FastAPI
# request handlers.
