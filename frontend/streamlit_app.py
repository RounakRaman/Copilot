"""
Streamlit frontend for the AI Job Application Copilot backend.

This is a thin UI layer over the FastAPI service deployed on Render. It
does not contain any AI logic itself -- everything (profile storage,
answer generation, OCR, form autofill) happens on the backend. This app
just makes those endpoints easy to use without going through /docs.

Configuration:
    Set BACKEND_URL below, or as a Streamlit secret / environment
    variable, to point at your deployed Render service.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud:
    1. Push this file to a GitHub repo (can be the same repo as the
       backend, or a separate one).
    2. Go to https://share.streamlit.io, connect the repo, point it at
       this file.
    3. In the app's "Secrets" settings, add:
           BACKEND_URL = "https://copilot-v2.onrender.com"
"""

import os
import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BACKEND_URL = "https://copilot-v2.onrender.com"


def get_backend_url() -> str:
    """
    Resolve the backend URL from, in order of priority:
    1. Streamlit secrets (st.secrets["BACKEND_URL"]) -- used when deployed
       on Streamlit Cloud.
    2. Environment variable BACKEND_URL -- used for local runs / other
       hosts.
    3. The hardcoded default above.
    """
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"]
    except Exception:
        pass
    return os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL)


BACKEND_URL = get_backend_url()
REQUEST_TIMEOUT = 60  # seconds; Render free tier cold-starts can be slow

st.set_page_config(page_title="Job Application Copilot", page_icon="📋", layout="centered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def backend_get(path: str):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def backend_post_json(path: str, payload: dict):
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "") if e.response is not None else ""
        except Exception:
            pass
        return None, f"{e}" + (f" -- {detail}" if detail else "")


def backend_post_form(path: str, data: dict, files: dict = None):
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", data=data, files=files, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "") if e.response is not None else ""
        except Exception:
            pass
        return None, f"{e}" + (f" -- {detail}" if detail else "")


# ---------------------------------------------------------------------------
# Sidebar -- backend status + config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Backend")
    st.caption(BACKEND_URL)

    if st.button("Check connection"):
        with st.spinner("Pinging backend..."):
            data, err = backend_get("/profile")
        if err:
            st.error(
                "Could not reach the backend. If it's been idle, Render free "
                "tier spins it down -- the first request after a while can "
                "take 30-60s to wake up. Try again.\n\n"
                f"Details: {err}"
            )
        else:
            st.success("Backend is reachable.")

    st.divider()
    st.caption(
        "Note: Render free tier disk is ephemeral. Profile data and "
        "uploaded resumes are not guaranteed to persist across backend "
        "restarts or redeploys."
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_profile, tab_answer, tab_autofill = st.tabs(["Profile", "Generate Answer", "Autofill Form"])


# ---- Profile tab ----------------------------------------------------------

with tab_profile:
    st.header("Your Profile")
    st.caption("Saved once on the backend. Used to autofill repetitive fields.")

    existing, err = backend_get("/profile")
    if err:
        st.warning(f"Could not load existing profile (backend may be waking up): {err}")
        existing = {}

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full name", value=existing.get("name") or "")
            email = st.text_input("Email", value=existing.get("email") or "")
            phone = st.text_input("Phone", value=existing.get("phone") or "")
            location = st.text_input("Location", value=existing.get("location") or "")
            linkedin = st.text_input("LinkedIn URL", value=existing.get("linkedin") or "")
            portfolio = st.text_input("Portfolio URL", value=existing.get("portfolio") or "")
            github = st.text_input(
                "Resume server path (advanced)",
                value=existing.get("github") or "",
                help=(
                    "Only set this if your resume already exists as a file "
                    "on the backend server itself. For uploading a resume "
                    "from this UI, use the Autofill tab instead."
                ),
            )
        with col2:
            current_company = st.text_input("Current company", value=existing.get("current_company") or "")
            current_role = st.text_input("Current role", value=existing.get("current_role") or "")
            experience_years = st.text_input("Years of experience", value=existing.get("experience_years") or "")
            notice_period = st.text_input("Notice period", value=existing.get("notice_period") or "")
            current_ctc = st.text_input("Current CTC", value=existing.get("current_ctc") or "")
            expected_ctc = st.text_input("Expected CTC", value=existing.get("expected_ctc") or "")
            work_authorization = st.text_input("Work authorization", value=existing.get("work_authorization") or "")

        skills_str = st.text_area(
            "Skills (comma-separated)",
            value=", ".join(existing.get("skills") or []),
        )
        education_str = st.text_area(
            "Education (one per line)",
            value="\n".join(existing.get("education") or []),
        )

        submitted = st.form_submit_button("Save profile", type="primary")

    if submitted:
        payload = {
            "name": name or None,
            "email": email or None,
            "phone": phone or None,
            "location": location or None,
            "linkedin": linkedin or None,
            "portfolio": portfolio or None,
            "github": github or None,
            "current_company": current_company or None,
            "current_role": current_role or None,
            "experience_years": experience_years or None,
            "notice_period": notice_period or None,
            "current_ctc": current_ctc or None,
            "expected_ctc": expected_ctc or None,
            "work_authorization": work_authorization or None,
            "skills": [s.strip() for s in skills_str.split(",") if s.strip()] or None,
            "education": [e.strip() for e in education_str.splitlines() if e.strip()] or None,
        }
        with st.spinner("Saving..."):
            result, err = backend_post_json("/profile", payload)
        if err:
            st.error(f"Failed to save profile: {err}")
        else:
            st.success("Profile saved.")


# ---- Generate Answer tab --------------------------------------------------

with tab_answer:
    st.header("Generate a Tailored Answer")
    st.caption("For free-form application questions, e.g. \"Why do you want to work here?\"")

    question = st.text_area("Question", placeholder="Why do you want to work here?")
    job_description = st.text_area(
        "Job description (optional)",
        placeholder="Paste the job description here for a more tailored answer.",
        height=150,
    )
    resume_path_for_answer = st.text_input(
        "Resume server path (optional)",
        help="A path to a PDF that already exists on the backend's disk. Leave blank to use the profile's saved path, if any.",
    )
    max_tokens = st.slider("Max answer length (tokens, approx.)", 50, 500, 200, step=50)

    if st.button("Generate answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            payload = {
                "question": question,
                "job_description": job_description or None,
                "resume_path": resume_path_for_answer or None,
                "max_tokens": max_tokens,
            }
            with st.spinner("Generating... (first request after idle may take a bit)"):
                result, err = backend_post_json("/generate-answer", payload)
            if err:
                st.error(f"Failed to generate answer: {err}")
            else:
                st.text_area("Answer", value=result.get("answer", ""), height=200)


# ---- Autofill tab ----------------------------------------------------------

with tab_autofill:
    st.header("Autofill a Job Application Form")
    st.caption(
        "Opens the form with a headless browser on the backend and fills in "
        "what it can. It never submits the form -- you review and submit "
        "yourself."
    )
    st.warning(
        "This automates a real browser against a real website. Test on a "
        "form you don't mind experimenting with before using it on an "
        "application that matters.",
        icon="⚠️",
    )

    url = st.text_input("Job application form URL", placeholder="https://...")

    resume_pdf = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    screenshot_img = st.file_uploader(
        "Screenshot of job description (optional)",
        type=["png", "jpg", "jpeg"],
        help="If the job description on the page is an image rather than text, upload a screenshot and OCR will extract it.",
    )

    if st.button("Run autofill", type="primary"):
        if not url.strip():
            st.warning("Enter a form URL first.")
        else:
            data = {"url": url}
            files = {}
            if resume_pdf is not None:
                files["resume_file"] = (resume_pdf.name, resume_pdf.getvalue(), "application/pdf")
            if screenshot_img is not None:
                files["screenshot"] = (screenshot_img.name, screenshot_img.getvalue(), screenshot_img.type)

            with st.spinner("Filling form... this can take 15-30s, longer if the backend was idle."):
                start = time.time()
                result, err = backend_post_form("/autofill", data=data, files=files or None)
                elapsed = time.time() - start

            if err:
                st.error(f"Autofill failed: {err}")
            else:
                st.success(f"Done in {elapsed:.1f}s. Review the filled fields below, then go submit manually.")
                st.json(result)
