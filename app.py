"""ResumeIQ — AI-powered resume parsing and candidate screening (Streamlit UI).

All screening logic lives in main.py. This file collects input, calls the
pipeline, and displays the results.

Run with:  uv run streamlit run app.py
"""

import hashlib
import html
import io
import json
import logging
import os

import streamlit as st

import main as pipeline
from ui_helpers import (
    MAX_FILE_BYTES,
    MAX_FILES,
    compose_job_text,
    candidate_label,
    estimate_minimum_seconds,
    format_size,
    friendly_error,
    humanize_key,
    matching_skills,
    missing_skills,
    find_text,
    normalize_results,
    results_to_csv,
    results_to_rows,
    skill_check,
    summary_metrics,
    unique_name,
    validate_upload,
)

logger = logging.getLogger("resumeiq.app")

st.set_page_config(page_title="ResumeIQ", page_icon="📄", layout="wide")

NAV = ["Dashboard", "Screen Resumes", "Candidate Results", "About the Project"]

# --------------------------------------------------------------------------
# Styling (uses Streamlit theme variables so light and dark mode both work)
# --------------------------------------------------------------------------

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.riq-brand { display:flex; align-items:center; gap:.65rem; margin-bottom:.2rem; }
.riq-brand-name { font-size:1.3rem; font-weight:700; letter-spacing:-.01em; line-height:1.1; }
.riq-brand-sub { font-size:.78rem; opacity:.7; line-height:1.3; margin-bottom:1rem; }
.riq-title { font-size:1.9rem; font-weight:700; letter-spacing:-.02em; margin:0 0 .15rem 0; padding:0; }
.riq-subtitle { opacity:.75; margin:0 0 1.4rem 0; }
.riq-hero { border:1px solid rgba(128,128,128,.28); border-radius:14px; padding:1.5rem 1.75rem;
            background:var(--secondary-background-color, rgba(128,128,128,.06)); margin-bottom:1.1rem; }
.riq-hero-title { font-size:1.35rem; font-weight:650; letter-spacing:-.01em; margin-bottom:.4rem; }
.riq-hero p { margin:0; opacity:.85; max-width:46rem; }
.riq-step { display:flex; align-items:center; gap:.6rem; margin-bottom:.35rem; }
.riq-stepnum { background:#4f46e5; color:#fff; width:1.5rem; height:1.5rem; border-radius:50%;
               display:inline-flex; align-items:center; justify-content:center; font-size:.8rem; font-weight:700; }
.riq-steptitle { font-size:1.1rem; font-weight:650; }
.riq-badge { display:inline-block; padding:.12rem .6rem; margin:.15rem .25rem .15rem 0; border-radius:999px;
             font-size:.8rem; font-weight:500; border:1px solid var(--b, rgba(128,128,128,.4));
             background:var(--bg, rgba(128,128,128,.12)); }
.riq-badge.green { --b: rgba(22,163,74,.6);  --bg: rgba(22,163,74,.16); }
.riq-badge.amber { --b: rgba(217,119,6,.65); --bg: rgba(217,119,6,.16); }
.riq-badge.red   { --b: rgba(220,38,38,.6);  --bg: rgba(220,38,38,.14); }
.riq-badge.blue  { --b: rgba(79,70,229,.6);  --bg: rgba(79,70,229,.14); }
.riq-chips { margin:.2rem 0 .6rem 0; }
.riq-muted { opacity:.7; font-size:.85rem; }
[data-testid="stMetricValue"] { font-size:1.7rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

BRAND = """
<div class="riq-brand">
  <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">
    <rect width="34" height="34" rx="9" fill="#4f46e5"/>
    <path d="M11 8h8l5 5v13a1 1 0 0 1-1 1H11a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"
          fill="none" stroke="#fff" stroke-width="1.8" stroke-linejoin="round"/>
    <path d="M19 8v5h5M13 18h8M13 22h6" stroke="#fff" stroke-width="1.8"
          stroke-linecap="round" fill="none"/>
  </svg>
  <div class="riq-brand-name">ResumeIQ</div>
</div>
<div class="riq-brand-sub">AI-powered resume parsing and candidate screening</div>
"""

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

DEFAULTS = {
    "nav": "Dashboard",
    "job_title": "",
    "company": "",
    "job_description": "",
    "required_skills": "",
    "preferred_skills": "",
    "min_exp": 0.0,
    "delay_seconds": pipeline.RATE_LIMIT_DELAY_SECONDS,
    "uploader_n": 0,
    "staged": {},  # unique file name -> {"data", "size", "digest"}
    "upload_notes": [],
    "results": None,  # None = no run yet; [] = run finished with no successes
    "failures": [],
    "job": None,
    "run_signature": None,
}
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)

# Widgets on a page that is not rendered lose their state in Streamlit.
# Re-assigning the value keeps the form intact when switching pages.
FORM_KEYS = [
    "job_title", "company", "job_description", "required_skills",
    "preferred_skills", "min_exp", "delay_seconds",
]
for _key in FORM_KEYS:
    st.session_state[_key] = st.session_state[_key]


def go(page: str) -> None:
    """Navigation callback (runs before the next script run)."""
    st.session_state["nav"] = page


def clear_session_data() -> None:
    for key in ("results", "job", "run_signature"):
        st.session_state[key] = None
    st.session_state["failures"] = []
    st.session_state["staged"] = {}
    st.session_state["upload_notes"] = []
    st.session_state["uploader_n"] += 1


def remove_file(name: str) -> None:
    st.session_state["staged"].pop(name, None)


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------


def api_key_ready() -> tuple[bool, str]:
    """Check GROQ_API_KEY (env, .env, or Streamlit secrets) without showing it."""
    message = (
        "GROQ_API_KEY is not configured. Add it to a local .env file, or to the "
        "app's secrets when deployed."
    )
    try:
        pipeline.get_client()
        return True, ""
    except ValueError:
        pass
    except Exception:
        return False, "The Groq client could not be initialised."

    try:
        secret = st.secrets["GROQ_API_KEY"]
    except Exception:
        return False, message
    if not secret:
        return False, message
    os.environ["GROQ_API_KEY"] = str(secret)
    try:
        pipeline.get_client()
        return True, ""
    except Exception:
        return False, message


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<h1 class="riq-title">{title}</h1><p class="riq-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def step_header(number: int, title: str, help_text: str = "") -> None:
    st.markdown(
        f'<div class="riq-step"><span class="riq-stepnum">{number}</span>'
        f'<span class="riq-steptitle">{title}</span></div>',
        unsafe_allow_html=True,
    )
    if help_text:
        st.caption(help_text)


def badge(text, kind: str = "gray") -> str:
    return f'<span class="riq-badge {kind}">{html.escape(str(text))}</span>'


def chips(items: list[str], kind: str = "gray", limit: int = 40) -> None:
    shown = items[:limit]
    markup = " ".join(badge(item, kind) for item in shown)
    if len(items) > limit:
        markup += " " + badge(f"+{len(items) - limit} more")
    st.markdown(f'<div class="riq-chips">{markup}</div>', unsafe_allow_html=True)


def score_kind(score: float) -> str:
    """Colour band for the score badge (a visual aid, not a decision threshold)."""
    if score >= 70:
        return "green"
    if score >= 40:
        return "amber"
    return "red"


def show_table(data, **kwargs) -> None:
    """st.dataframe stretched to full width across Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def empty_state(title: str, body: str, button_label: str, target: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(body)
        st.button(button_label, type="primary", on_click=go, args=(target,),
                  key=f"empty_{target}")


def failure_list(failures: list[dict]) -> None:
    for failure in failures:
        st.error(f"**{failure['file']}** — {failure['reason']}")


# --------------------------------------------------------------------------
# Upload handling
# --------------------------------------------------------------------------


def ingest_uploads(uploads) -> None:
    """Validate new uploads and add them to the session's staged files."""
    staged = st.session_state["staged"]
    known_digests = {info["digest"]: name for name, info in staged.items()}
    notes: list[tuple[str, str]] = []

    for upload in uploads:
        data = upload.getvalue()
        name = upload.name

        if len(staged) >= MAX_FILES:
            notes.append(("warning", f"{name}: skipped, the limit of {MAX_FILES} files was reached."))
            continue
        reason = validate_upload(name, data)
        if reason:
            notes.append(("error", f"{name}: {reason}"))
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in known_digests:
            notes.append(("info", f"{name}: skipped, identical to {known_digests[digest]}."))
            continue
        final_name = unique_name(name, set(staged))
        if final_name != name:
            notes.append(("info", f"{name}: renamed to {final_name} because another file uses that name."))
        staged[final_name] = {"data": data, "size": len(data), "digest": digest}
        known_digests[digest] = final_name

    st.session_state["upload_notes"] = notes


def run_signature(job_text: str, delay: float) -> str:
    parts = [job_text, str(delay)]
    parts += sorted(f"{name}:{info['digest']}" for name, info in st.session_state["staged"].items())
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Screening run (calls the pipeline in main.py)
# --------------------------------------------------------------------------


def run_screening(job_text: str, signature: str) -> None:
    delay = st.session_state["delay_seconds"]
    progress = st.progress(0.0, text="Reading the job description...")
    counts = st.empty()

    try:
        job = pipeline.parse_job(job_text)
    except Exception as exc:
        logger.warning("Job description parsing failed (%s)", type(exc).__name__)
        progress.empty()
        st.error(
            "The job description could not be analyzed. "
            f"{friendly_error(exc)} Your inputs are still here, so you can try again."
        )
        return

    files = []
    for name, info in st.session_state["staged"].items():
        buffer = io.BytesIO(info["data"])
        buffer.name = name
        files.append(buffer)

    def on_progress(done: int, total: int, current: str | None, ok: int, failed: int) -> None:
        if current:
            progress.progress(done / total, text=f"Processing {current} ({done + 1} of {total})")
        else:
            progress.progress(1.0, text="Finished")
        counts.markdown(
            f"Succeeded: **{ok}** &nbsp;·&nbsp; Failed: **{failed}** "
            f"&nbsp;·&nbsp; Remaining: **{total - done}**"
        )

    try:
        results, failures = pipeline.screen_resumes(
            job, files, delay_seconds=delay, on_progress=on_progress
        )
    except Exception as exc:
        logger.warning("Screening stopped unexpectedly (%s)", type(exc).__name__)
        progress.empty()
        st.error(f"Screening stopped unexpectedly. {friendly_error(exc)}")
        return

    st.session_state["results"] = normalize_results(results)
    st.session_state["failures"] = [
        {"file": failure["file"], "reason": friendly_error(failure["error"])}
        for failure in failures
    ]
    st.session_state["job"] = job.model_dump()
    st.session_state["run_signature"] = signature

    if results:
        st.success(
            f"Screening complete: {len(results)} resume(s) evaluated"
            + (f", {len(failures)} failed." if failures else ".")
        )
        st.button("View candidate results", type="primary", on_click=go,
                  args=("Candidate Results",), key="after_run_view")
    else:
        st.warning("No resumes could be processed. See the reasons below.")
    if failures:
        failure_list(st.session_state["failures"])


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def page_dashboard() -> None:
    page_header("Dashboard", "Overview of your current screening session.")

    st.markdown(
        '<div class="riq-hero"><div class="riq-hero-title">Screen resumes faster, '
        "with reasons you can check.</div>"
        "<p>ResumeIQ reads PDF and DOCX resumes, extracts structured candidate "
        "information, and compares each candidate with your job description. "
        "Results are a starting point for a human reviewer, never an automatic "
        "hiring decision.</p></div>",
        unsafe_allow_html=True,
    )

    results = st.session_state["results"]
    failures = st.session_state["failures"]

    if results is None:
        empty_state(
            "No screening yet",
            "Add a job description and some resumes to get ranked candidates. "
            "Your metrics will appear here once a screening has run.",
            "Start screening",
            "Screen Resumes",
        )
        cols = st.columns(3)
        steps = [
            ("1. Describe the role", "Paste the job description. Skills and experience are extracted from it."),
            ("2. Upload resumes", "Add PDF or DOCX files. Each is validated before it is processed."),
            ("3. Review candidates", "See scores with the model's reasoning, and export to CSV."),
        ]
        for column, (title, body) in zip(cols, steps):
            with column, st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(body)
        return

    metrics = summary_metrics(results, failures)
    cols = st.columns(4)
    values = [
        ("Resumes processed", metrics["total_attempted"]),
        ("Candidates evaluated", metrics["processed_ok"]),
        ("Average match score", f"{metrics['average']:.1f}" if metrics["average"] is not None else "—"),
        ("Failed files", metrics["failed"]),
    ]
    for column, (label, value) in zip(cols, values):
        with column, st.container(border=True):
            st.metric(label, value)

    job = st.session_state["job"] or {}
    if job.get("role"):
        st.caption(f"Last screening was against: {job['role']}")

    left, right, _ = st.columns([2, 2, 6])
    left.button("View results", type="primary", on_click=go, args=("Candidate Results",), key="dash_results")
    right.button("New screening", on_click=go, args=("Screen Resumes",), key="dash_new")


def page_screen() -> None:
    page_header("Screen resumes", "Describe the role, add resumes, and analyze candidates.")

    key_ready, key_message = api_key_ready()
    if not key_ready:
        st.warning(key_message)

    # Step 1 -----------------------------------------------------------------
    with st.container(border=True):
        step_header(1, "Job description",
                    "Only the description is required. Any optional field you fill in is added to it before analysis.")
        col_a, col_b = st.columns(2)
        col_a.text_input("Job title (optional)", key="job_title", placeholder="e.g. Software Development Engineer")
        col_b.text_input("Company (optional)", key="company")
        st.text_area("Job description", key="job_description", height=240,
                     placeholder="Paste the full job description: responsibilities, required and preferred qualifications...")
        col_c, col_d, col_e = st.columns(3)
        col_c.text_input("Required skills (optional)", key="required_skills",
                         help="Comma-separated. Leave blank to extract them from the job description.")
        col_d.text_input("Preferred skills (optional)", key="preferred_skills", help="Comma-separated.")
        col_e.number_input("Minimum experience, years (optional)", min_value=0.0, max_value=50.0,
                           step=0.5, key="min_exp", help="0 means not specified.")

    # Step 2 -----------------------------------------------------------------
    with st.container(border=True):
        step_header(2, "Resume upload",
                    f"PDF or DOCX, up to {format_size(MAX_FILE_BYTES)} each and {MAX_FILES} files per session. "
                    "Scanned or image-only PDFs contain no readable text (OCR is not supported) "
                    "and will be reported as failed.")
        uploads = st.file_uploader(
            "Add resumes", type=["pdf", "docx"], accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_n']}", label_visibility="collapsed",
        )
        if uploads:
            ingest_uploads(uploads)
            st.session_state["uploader_n"] += 1
            st.rerun()

        for level, note in st.session_state["upload_notes"]:
            getattr(st, level)(note)
        st.session_state["upload_notes"] = []

        staged = st.session_state["staged"]
        if staged:
            st.markdown(f"**{len(staged)} resume(s) ready**")
            for name, info in list(staged.items()):
                col_name, col_size, col_remove = st.columns([6, 2, 2])
                col_name.write(name)
                col_size.caption(format_size(info["size"]))
                col_remove.button("Remove", key=f"rm_{name}", on_click=remove_file, args=(name,))
        else:
            st.caption("No resumes added yet.")

    # Step 3 -----------------------------------------------------------------
    with st.container(border=True):
        step_header(3, "Screening configuration")
        st.slider("Pause between API calls (seconds)", 0, 15, key="delay_seconds",
                  help="Each resume needs two model calls. A longer pause lowers the chance of hitting Groq's rate limit.")
        wait = estimate_minimum_seconds(len(st.session_state["staged"]), st.session_state["delay_seconds"])
        st.caption(
            f"Time spent pausing alone: about {wait:.0f} s for {len(st.session_state['staged'])} resume(s); "
            "the model calls themselves add to that."
        )
        st.caption(
            "Scoring weights are not configurable yet. The match score comes from the model's overall "
            "comparison of the job and the resume (see About the Project)."
        )

    # Step 4 -----------------------------------------------------------------
    with st.container(border=True):
        step_header(4, "Analyze candidates")

        description = st.session_state["job_description"]
        staged = st.session_state["staged"]
        problems = []
        if not description.strip():
            problems.append("Add a job description in step 1.")
        if not staged:
            problems.append("Add at least one resume in step 2.")
        if not key_ready:
            problems.append("Configure GROQ_API_KEY.")
        for problem in problems:
            st.info(problem)

        job_text = compose_job_text(
            st.session_state["job_title"], st.session_state["company"], description,
            st.session_state["required_skills"], st.session_state["preferred_skills"],
            st.session_state["min_exp"],
        )
        signature = run_signature(job_text, st.session_state["delay_seconds"])
        unchanged = st.session_state["run_signature"] == signature and not problems
        force = False
        if unchanged:
            st.info("These exact inputs were already analyzed, so no new API calls are needed. "
                    "Open Candidate Results, or change the inputs to run again.")
            force = st.checkbox("Run again anyway (uses API calls)", key="force_rerun")

        if st.button("Analyze candidates", type="primary", disabled=bool(problems) or (unchanged and not force)):
            run_screening(job_text, signature)


def render_value(value) -> None:
    """Generic display for the model's free-form evaluation details."""
    if isinstance(value, list):
        for item in value:
            st.markdown(f"- {item}" if not isinstance(item, (dict, list)) else f"- `{json.dumps(item)}`")
    elif isinstance(value, dict):
        for key, item in value.items():
            text = item if not isinstance(item, (dict, list)) else f"`{json.dumps(item)}`"
            st.markdown(f"- **{humanize_key(key)}:** {text}")
    elif isinstance(value, bool):
        st.write("Yes" if value else "No")
    else:
        st.write(value)


def render_candidate(result: dict) -> None:
    resume = result.get("resume") or {}
    details = result.get("details") or {}
    job = st.session_state["job"] or {}
    score = result.get("score_valid")

    head_left, head_right = st.columns([4, 1])
    head_left.markdown(f"### {candidate_label(result)}")
    head_left.caption(result["file"])
    if score is not None:
        head_right.markdown(badge(f"Match score {score:.0f}/100", score_kind(score)), unsafe_allow_html=True)
    else:
        head_right.markdown(badge("Score unavailable", "red"), unsafe_allow_html=True)

    tab_profile, tab_skills, tab_exp, tab_match, tab_file = st.tabs(
        ["Profile", "Skills", "Experience", "Match explanation", "Original resume"]
    )

    with tab_profile:
        def field(label, value):
            st.markdown(f"**{label}:** {value if value not in (None, '', []) else 'Not found in resume'}")

        field("Name", resume.get("name"))
        years = resume.get("total_experience_years")
        field("Total experience", f"about {years:g} years (as parsed from the resume)" if years is not None else None)
        for label, key in (("Location", "location"), ("Summary", "summary"), ("Languages", "languages"), ("Links", "links")):
            if resume.get(key):  # shown only if the parsed resume has the field
                value = resume[key]
                field(label, ", ".join(map(str, value)) if isinstance(value, list) else value)
        education = resume.get("education") or []
        st.markdown("**Education:**" + ("" if education else " Not found in resume"))
        for item in education:
            st.markdown(f"- {item}")
        if st.checkbox("Show contact details", key=f"contact_{result['file']}"):
            field("Email", resume.get("email"))
            field("Phone", resume.get("phone"))

    with tab_skills:
        st.caption(
            "Exact-text check of the job's skills against the skills parsed from this resume. "
            "'Not found' only means the skill was not listed. It does not prove the candidate lacks it, "
            "and synonyms (for example JS and JavaScript) are not matched."
        )
        for title, key in (("Required skills", "required_skills"), ("Preferred skills", "preferred_skills")):
            job_skills = job.get(key) or []
            st.markdown(f"**{title}**")
            if not job_skills:
                st.caption("None identified in the job description.")
                continue
            found, unverified = skill_check(job_skills, resume)
            if found:
                st.caption("Found in the resume's skills")
                chips(found, "green")
            if unverified:
                st.caption("Not found in listed skills: verify manually")
                chips(unverified, "amber")

        model_matches, model_gaps = matching_skills(details), missing_skills(details)
        if model_matches or model_gaps:
            st.markdown("**Reported by the model's evaluation**")
            if model_matches:
                st.caption("Matching")
                chips(model_matches, "green")
            if model_gaps:
                st.caption("Gaps or unverified")
                chips(model_gaps, "amber")
        all_skills = resume.get("skills") or []
        if all_skills:
            st.markdown("**All skills parsed from the resume**")
            chips(all_skills)

    with tab_exp:
        experiences = resume.get("experiences") or []
        if not experiences:
            st.info("No work experience was found in the parsed resume.")
        for experience in experiences:
            with st.container(border=True):
                role = experience.get("role") or "Role not stated"
                company = experience.get("company")
                st.markdown(f"**{role}**" + (f" at {company}" if company else ""))
                if experience.get("duration"):
                    st.caption(experience["duration"])
                if experience.get("description"):
                    st.write(experience["description"])
                if experience.get("skills_used"):
                    chips(experience["skills_used"])
        projects = resume.get("projects") or []
        if projects:
            st.markdown("**Projects**")
            for project in projects:
                st.markdown(f"- {project}")
        certifications = resume.get("certifications") or []
        if certifications:
            st.markdown("**Certifications**")
            for certification in certifications:
                st.markdown(f"- {certification}")
        st.caption("Listed as parsed from the resume. Relevance to the role is part of the model's evaluation in the next tab.")

    with tab_match:
        if score is not None:
            st.progress(score / 100, text=f"Match score: {score:.0f} / 100")
        else:
            st.warning("The model returned a score outside 0-100, so this candidate is left unranked.")
        verdict = find_text(details)
        if verdict:
            st.info(verdict)
        for key, value in details.items():
            st.markdown(f"**{humanize_key(key)}**")
            render_value(value)
        st.caption("This evaluation is generated by a language model. Check it against the original resume before acting on it.")

    with tab_file:
        info = st.session_state["staged"].get(result["file"])
        st.write(result["file"])
        if info:
            mime = "application/pdf" if result["file"].lower().endswith(".pdf") else (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            st.download_button("Download original", data=info["data"], file_name=result["file"], mime=mime,
                               key=f"dl_{result['file']}")
            st.caption("Available only while the file stays in this session.")
        else:
            st.info("The original file is no longer in this session (it was removed or the session data was cleared).")


def page_results() -> None:
    page_header("Candidate results", "Ranked candidates with the reasoning behind each score.")

    results = st.session_state["results"]
    failures = st.session_state["failures"]

    if results is None:
        empty_state("No results yet", "Run a screening to see ranked candidates here.",
                    "Go to Screen Resumes", "Screen Resumes")
        return

    metrics = summary_metrics(results, failures)
    cols = st.columns(4)
    values = [
        ("Processed successfully", metrics["processed_ok"]),
        ("Failed files", metrics["failed"]),
        ("Average match score", f"{metrics['average']:.1f}" if metrics["average"] is not None else "—"),
        ("Highest match score", f"{metrics['highest']:.0f}" if metrics["highest"] is not None else "—"),
    ]
    for column, (label, value) in zip(cols, values):
        with column, st.container(border=True):
            st.metric(label, value)

    st.caption("Scores come from a language model and are a starting point for human review, "
               "not a hiring decision. Colour bands are a visual aid only.")
    if metrics["invalid_scores"]:
        st.warning(f"{metrics['invalid_scores']} result(s) returned a score outside 0-100 and were left unranked.")

    if failures:
        with st.expander(f"{len(failures)} file(s) could not be processed", expanded=not results):
            failure_list(failures)
    if not results:
        return

    # Filters and sorting ----------------------------------------------------
    with st.container(border=True):
        col_min, col_search, col_sort = st.columns([2, 3, 3])
        min_score = col_min.slider("Minimum match score", 0, 100, 0)
        query = col_search.text_input("Search candidate or file name").strip().lower()
        sort_by = col_sort.selectbox("Sort by", [
            "Match score (high to low)", "Match score (low to high)",
            "Candidate name (A to Z)", "Experience (most years first)",
        ])

    rows = results_to_rows(results)
    pairs = list(zip(results, rows))
    pairs = [
        (res, row) for res, row in pairs
        if (min_score == 0 or (row["Match score"] is not None and row["Match score"] >= min_score))
        and (not query or query in str(row["Candidate"]).lower() or query in str(row["Resume file"]).lower())
    ]

    def sort_key(pair):
        _, row = pair
        if sort_by == "Candidate name (A to Z)":
            return str(row["Candidate"]).lower()
        if sort_by == "Experience (most years first)":
            years = row["Experience (years, as parsed)"]
            return -(years if years is not None else -1)
        score = row["Match score"]
        if sort_by == "Match score (low to high)":
            return score if score is not None else 101
        return -(score if score is not None else -1)

    pairs.sort(key=sort_key)

    if not pairs:
        st.info("No candidates match the current filters.")
    else:
        table = [
            {
                "Rank": row["Rank"],
                "Candidate": row["Candidate"],
                "Resume file": row["Resume file"],
                "Match score": row["Match score"],
                "Experience (yrs)": row["Experience (years, as parsed)"],
                "Matching skills": len(row["Matching skills"]) if row["Matching skills"] is not None else None,
                "Missing / unverified": len(row["Missing or unverified skills"])
                if row["Missing or unverified skills"] is not None else None,
            }
            for _, row in pairs
        ]
        show_table(
            table,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d", width="small"),
                "Match score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                "Experience (yrs)": st.column_config.NumberColumn(format="%.1f"),
                "Matching skills": st.column_config.NumberColumn(format="%d"),
                "Missing / unverified": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption(
            "Experience and skill counts are shown only when the resume or the model's evaluation provided them. "
            "Click a column header to sort."
        )

    st.download_button("Export all results (CSV)", data=results_to_csv(results),
                       file_name="resumeiq_results.csv", mime="text/csv")

    # Candidate details ------------------------------------------------------
    if pairs:
        st.divider()
        labels = {
            res["file"]: f"{'#' + str(res['rank']) if res['rank'] else '—'}  {candidate_label(res)}"
            for res, _ in pairs
        }
        chosen = st.selectbox("Select a candidate to review", [res["file"] for res, _ in pairs],
                              format_func=labels.get)
        render_candidate(next(res for res, _ in pairs if res["file"] == chosen))


def page_about() -> None:
    page_header("About the project", "How ResumeIQ works, what it does with data, and what it cannot do yet.")

    with st.container(border=True):
        st.markdown("#### How screening works")
        st.markdown(
            f"""
1. **Job description.** Your form fields and the pasted description are combined and parsed by a language model
   (`{pipeline.MODEL}` on Groq) into a role, required and preferred skills, minimum experience, education
   requirements and responsibilities.
2. **Resumes.** Text is extracted from each PDF or DOCX (pypdf, python-docx), then parsed by the model into name,
   contact details, skills, experience, education, projects and certifications. The prompts tell the model not to
   invent values, but model output is not guaranteed to be correct.
3. **Score.** The model compares the structured job and resume and returns a score from 0 to 100 with a free-form
   explanation. The app checks the score is within 0-100, leaves out-of-range scores unranked, and sorts by score.
   There is no separate rule-based scoring or weighting yet.
"""
        )

    with st.container(border=True):
        st.markdown("#### Data handling")
        st.markdown(
            """
- Resume text and the parsed fields are sent to Groq's API for processing.
- Uploaded files and results are kept in the memory of this app's session only. The app does not write them to disk
  or to a database.
- Remove files one by one in step 2, or use **Clear session data** in the sidebar. This clears them from the app
  session. It does not affect copies held by Groq or by the hosting platform, which this app cannot control.
- Contact details are hidden until you choose to show them and are not included in the CSV export.
"""
        )

    with st.container(border=True):
        st.markdown("#### Limitations")
        st.markdown(
            """
- Scanned or image-only PDFs have no extractable text (no OCR), and text inside DOCX tables is not read.
- The evaluation details are free-form model output, so skill counts and summaries appear only when the model returns them.
- The skill check on the Skills tab is exact text matching and does not understand synonyms.
- Names and other details can act as proxies for protected characteristics. Always review the results yourself.
- Requests run one at a time with a pause between calls, so large batches take a while.
"""
        )

    with st.container(border=True):
        st.markdown("#### Planned (not available yet)")
        st.markdown(
            """
- Rule-based scoring with adjustable weights and evidence quotes from the resume
- Automatic retries with backoff for rate limits and network errors
- Richer parsing (location, summary, links, languages) and a required-skill filter
- PDF and Excel export
"""
        )

    st.caption("ResumeIQ supports human recruiters. It does not make hiring decisions.")


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(BRAND, unsafe_allow_html=True)
    st.radio("Navigation", NAV, key="nav", label_visibility="collapsed")
    st.divider()
    st.caption("Assists human reviewers. Not a hiring decision.")
    if st.session_state["results"] is not None or st.session_state["staged"]:
        st.button("Clear session data", on_click=clear_session_data)

PAGES = {
    "Dashboard": page_dashboard,
    "Screen Resumes": page_screen,
    "Candidate Results": page_results,
    "About the Project": page_about,
}
PAGES[st.session_state["nav"]]()
