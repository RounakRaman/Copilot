# AI Job Application Copilot

This repository contains a proof‑of‑concept “copilot” for automating job
applications. It brings together browser automation, OCR, resume
parsing and a lightweight language model into a single service that
reduces the tedium of filling out repetitive forms.

## Features

* **Static profile store:** Fill out your details once and save them in
  `profile.json`. The API exposes `/profile` for retrieving and
  updating this information.
* **Resume extraction:** Extracts text from PDF resumes using
  PyPDF2.
* **OCR support:** Reads job descriptions embedded in images via
  EasyOCR. This is useful when companies include their JD as an
  image in Google Forms or ATS portals.
* **Answer generation:** Uses Hugging Face’s `bart-large-cnn`
  summariser and `flan‑t5‑base` generator to craft personalised
  answers to open‑ended questions. Should you prefer to use a
  commercial API, swap out `ai_utils.generate_answer()`.
* **Form filling:** Launches a headless Chromium instance via
  Playwright, visits the application URL and heuristically matches
  inputs to profile fields. It fills in names, emails, phone
  numbers, LinkedIn URLs and uploads your resume. For free‑form
  questions (e.g. “Why do you want to work here?”) the service
  generates an answer from your resume and the job description.
* **Human in the loop:** The automation stops short of submitting the
  application. After populating the form it returns a report so
  you can review and click “Submit” yourself.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install  # installs browser binaries
```

Run the API server using Uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

You can then interact with the API at [http://localhost:8000/docs](http://localhost:8000/docs).

> **Note:** The first request to `/generate-answer` may take several
> seconds because the summarisation and generation models are loaded
> lazily. Subsequent calls will be faster.

## Hosting

This project is designed to run on any environment that can host a
Python web application. Many platforms provide a free tier suitable
for prototypes. According to a January 2026 roundup on Resourify,
developers can deploy full‑stack applications to services like
Vercel, Netlify, Cloudflare Pages, GitHub Pages and Render without
paying a cent【449156299932304†L49-L149】. Vercel and Netlify excel at
hosting frontend and serverless functions, while Render’s free tier
supports long‑running web services with automatic sleep on
inactivity【449156299932304†L144-L150】.

For more complex deployments you might choose Fly.io or Railway, but
these have transitioned to trial‑credit models【449156299932304†L161-L167】.
Deployment steps vary by provider; at a high level you will:

1. Create an account with your chosen platform (e.g. Vercel).
2. Push this repository to a Git provider (GitHub, GitLab).
3. Link the repository to the platform and configure the build
   command (`pip install -r requirements.txt` and `uvicorn`) and the
   start command (`uvicorn app.main:app --host 0.0.0.0 --port $PORT`).
4. For Playwright to work in a serverless environment you may need to
   choose a platform that supports persistent containers (e.g. Render
   or Fly.io) or use headless browserless services. Serverless
   offerings like Vercel and Netlify are ideal for the API only; you
   can run the automation component locally or on a separate
   worker.

## Free AI options

The AI models bundled here run locally. If you would rather use
cloud‑hosted models, several providers offer generous free access as
of April 2026. A developer cheat‑sheet notes that Google’s Gemini
API, via AI Studio, provides the most generous ongoing free tier
among the major providers【817524471171480†L89-L106】. It allows you to
experiment with the Gemini Flash and Pro models without a credit
card, albeit with lower rate limits than paid plans【817524471171480†L90-L103】. OpenAI and Anthropic
offer small initial credits, but these expire quickly【817524471171480†L50-L87】. If you choose to integrate
with these APIs, simply modify `ai_utils.generate_answer()` to call
the service instead of the local model.

## Next steps

This project is intentionally simple. To turn it into a polished
assistant you might consider:

* Improving field matching with machine learning or regular
  expressions rather than simple keyword searches.
* Persisting profiles in a database and supporting multiple users.
* Caching AI responses to avoid regenerating answers for the same
  question and context.
* Adding authentication and a frontend UI for managing applications.
* Building a Chrome extension that triggers the backend when a
  career page or Google Form loads, as described in the concept
  overview.

We hope this serves as a useful starting point for your own job
application copilot!