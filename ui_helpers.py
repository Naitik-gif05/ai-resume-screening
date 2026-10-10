"""Pure helper functions for the ResumeIQ interface.

Nothing here imports Streamlit, so everything can be unit tested directly.
Screening logic stays in main.py; this module only prepares inputs for the
pipeline and shapes its output for display and export.
"""

import csv
import io
import re

MAX_FILE_BYTES = 5 * 1024 * 1024  # per resume
MAX_FILES = 25  # per session (API calls are paced, so large batches are slow)
ALLOWED_EXTENSIONS = (".pdf", ".docx")

CSV_COLUMNS = [
    "Rank",
    "Candidate",
    "Resume file",
    "Match score",
    "Experience (years, as parsed)",
    "Matching skills",
    "Missing or unverified skills",
    "Evaluation summary",
]


# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------


def format_size(num_bytes: int) -> str:
    """Human-readable file size."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def validate_upload(name: str, data: bytes) -> str | None:
    """Return a user-facing reason if the upload should be rejected, else None."""
    lowered = name.lower()
    if not lowered.endswith(ALLOWED_EXTENSIONS):
        return "unsupported file type (use PDF or DOCX)."
    if not data:
        return "the file is empty."
    if len(data) > MAX_FILE_BYTES:
        return (
            f"file is {format_size(len(data))}; the limit is "
            f"{format_size(MAX_FILE_BYTES)} per resume."
        )
    if lowered.endswith(".pdf") and not data.lstrip()[:5].startswith(b"%PDF"):
        return "this does not look like a valid PDF."
    if lowered.endswith(".docx") and not data.startswith(b"PK"):
        return "this does not look like a valid DOCX."
    return None


def unique_name(name: str, existing: set[str]) -> str:
    """Return `name`, or 'name (2).ext', 'name (3).ext'... if already used."""
    if name not in existing:
        return name
    dot = name.rfind(".")
    stem, ext = (name[:dot], name[dot:]) if dot > 0 else (name, "")
    counter = 2
    while f"{stem} ({counter}){ext}" in existing:
        counter += 1
    return f"{stem} ({counter}){ext}"


# --------------------------------------------------------------------------
# Job description
# --------------------------------------------------------------------------


def split_skills(text: str) -> list[str]:
    """Split a comma / semicolon / newline separated skills string."""
    parts = re.split(r"[,;\n]", text or "")
    seen: set[str] = set()
    skills: list[str] = []
    for part in parts:
        skill = part.strip()
        if skill and skill.lower() not in seen:
            seen.add(skill.lower())
            skills.append(skill)
    return skills


def compose_job_text(
    title: str,
    company: str,
    description: str,
    required_skills: str,
    preferred_skills: str,
    min_years: float,
) -> str:
    """Combine the form fields into one text block for parse_job().

    Optional fields are only included when filled in, so the pasted job
    description alone is enough.
    """
    lines: list[str] = []
    if title.strip():
        lines.append(f"Job title: {title.strip()}")
    if company.strip():
        lines.append(f"Company: {company.strip()}")
    required = split_skills(required_skills)
    if required:
        lines.append("Required skills: " + ", ".join(required))
    preferred = split_skills(preferred_skills)
    if preferred:
        lines.append("Preferred skills: " + ", ".join(preferred))
    if min_years and min_years > 0:
        lines.append(f"Minimum experience: {min_years:g} years")
    header = "\n".join(lines)
    description = description.strip()
    return f"{header}\n\n{description}" if header else description


def estimate_minimum_seconds(num_files: int, delay_seconds: float) -> float:
    """Pause time alone: the pipeline sleeps after each of its two calls per file."""
    return max(0, num_files) * 2 * max(0.0, delay_seconds)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


def valid_score(value) -> float | None:
    """Return the score as a float if it is a number in 0..100, else None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or not 0 <= number <= 100:  # NaN or out of range
        return None
    return number


def _norm_key(key) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def humanize_key(key) -> str:
    return _norm_key(key).replace("_", " ").capitalize()


def find_list(details: dict, include: tuple[str, ...], exclude: tuple[str, ...] = ()):
    """Best-effort: first list value in `details` whose key contains an `include`
    word and none of the `exclude` words. Returns list[str] or None.

    The model's `details` is free-form, so this only recognises lists the
    model actually returned; it never invents entries.
    """
    if not isinstance(details, dict):
        return None
    for key, value in details.items():
        normalized = _norm_key(key)
        if any(word in normalized for word in include) and not any(
            word in normalized for word in exclude
        ):
            if isinstance(value, list):
                return [str(item) for item in value if str(item).strip()]
    return None


def matching_skills(details: dict):
    return find_list(
        details,
        include=("match",),
        exclude=("missing", "percent", "overall", "not_", "score"),
    )


def missing_skills(details: dict):
    return find_list(details, include=("missing", "gap", "not_found", "lack", "unverified"))


def find_text(details: dict, include: tuple[str, ...] = ("verdict", "summary", "recommendation")):
    """Best-effort: first non-empty string value whose key contains an `include` word."""
    if not isinstance(details, dict):
        return None
    for key, value in details.items():
        if any(word in _norm_key(key) for word in include) and isinstance(value, str):
            if value.strip():
                return value.strip()
    return None


def normalize_results(results: list[dict]) -> list[dict]:
    """Attach `score_valid` and `rank` to pipeline results.

    Scores outside 0..100 (or non-numeric) are flagged invalid: they get no
    rank and are excluded from averages. Ranking is by valid score, descending.
    """
    rows = []
    for result in results:
        row = dict(result)
        row["score_valid"] = valid_score(result.get("score"))
        rows.append(row)

    valid = sorted(
        (row for row in rows if row["score_valid"] is not None),
        key=lambda row: row["score_valid"],
        reverse=True,
    )
    invalid = [row for row in rows if row["score_valid"] is None]

    for position, row in enumerate(valid, start=1):
        row["rank"] = position
    for row in invalid:
        row["rank"] = None

    return valid + invalid


def candidate_label(row: dict) -> str:
    return row.get("name") or row.get("file") or "Unnamed candidate"


def summary_metrics(results: list[dict], failures: list[dict]) -> dict:
    """Metrics shown on the dashboard / results page (computed from real data)."""
    scores = [
        row["score_valid"] for row in results if row.get("score_valid") is not None
    ]
    return {
        "processed_ok": len(results),
        "failed": len(failures),
        "total_attempted": len(results) + len(failures),
        "average": sum(scores) / len(scores) if scores else None,
        "highest": max(scores) if scores else None,
        "invalid_scores": len(results) - len(scores),
    }


def skill_check(job_skills: list[str], resume: dict) -> tuple[list[str], list[str]]:
    """Exact-text (case-insensitive) check of job skills against resume skills.

    Returns (found, not_found_in_listed_skills). "Not found" means the skill
    was not in the parsed skill lists; it is NOT proof the candidate lacks it,
    so the UI labels these "needs manual verification".
    """
    available: set[str] = set()
    for skill in resume.get("skills") or []:
        available.add(str(skill).strip().lower())
    for experience in resume.get("experiences") or []:
        for skill in (experience or {}).get("skills_used") or []:
            available.add(str(skill).strip().lower())

    found, not_found = [], []
    for skill in job_skills:
        (found if skill.strip().lower() in available else not_found).append(skill)
    return found, not_found


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def _csv_safe(value) -> str:
    """Neutralise spreadsheet formula injection in text taken from resumes."""
    text = "" if value is None else str(value)
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text


def results_to_rows(results: list[dict]) -> list[dict]:
    """Flat rows (one per candidate) used by both the table and the CSV."""
    rows = []
    for result in results:
        details = result.get("details") or {}
        resume = result.get("resume") or {}
        rows.append(
            {
                "Rank": result.get("rank"),
                "Candidate": candidate_label(result),
                "Resume file": result.get("file"),
                "Match score": result.get("score_valid"),
                "Experience (years, as parsed)": resume.get("total_experience_years"),
                "Matching skills": matching_skills(details),
                "Missing or unverified skills": missing_skills(details),
                "Evaluation summary": find_text(details),
            }
        )
    return rows


def results_to_csv(results: list[dict]) -> str:
    """CSV text for all screened candidates (contact details are not included)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    for row in results_to_rows(results):
        writer.writerow(
            [
                row["Rank"] if row["Rank"] is not None else "",
                _csv_safe(row["Candidate"]),
                _csv_safe(row["Resume file"]),
                f"{row['Match score']:.1f}" if row["Match score"] is not None else "",
                row["Experience (years, as parsed)"]
                if row["Experience (years, as parsed)"] is not None
                else "",
                _csv_safe("; ".join(row["Matching skills"] or [])),
                _csv_safe("; ".join(row["Missing or unverified skills"] or [])),
                _csv_safe(row["Evaluation summary"]),
            ]
        )
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

_READABLE_ERRORS = {"ResumeExtractionError", "UnsupportedFileTypeError"}


def friendly_error(error) -> str:
    """Turn an exception or a 'ClassName: message' string into a safe user message.

    Raw exception text from unknown errors is not shown (it can contain
    request details or resume content).
    """
    if isinstance(error, BaseException):
        class_name, message = type(error).__name__, str(error)
    else:
        class_name, _, message = str(error).partition(": ")
    lowered = f"{class_name} {message}".lower()

    if "groq_api_key" in lowered:
        return (
            "The Groq API key is not configured. Add GROQ_API_KEY to your .env "
            "file or to the deployment secrets."
        )
    if class_name == "AuthenticationError" or "invalid api key" in lowered:
        return "Groq rejected the API key. Check that GROQ_API_KEY is correct and active."
    if class_name == "RateLimitError" or "rate limit" in lowered or "429" in lowered:
        return (
            "Groq's rate limit was reached. Wait a minute, increase the pause "
            "between requests, and run the analysis again."
        )
    if class_name in {"APIConnectionError", "APITimeoutError"}:
        return "Could not reach Groq (network problem or timeout). Check your connection and try again."
    if class_name == "LLMOutputError":
        return "The model returned a response that could not be read. Running the analysis again usually helps."
    if class_name in _READABLE_ERRORS and message:
        return message
    if class_name == "InternalServerError" or "503" in lowered or "502" in lowered:
        return "Groq is temporarily unavailable. Try again in a moment."
    return f"Unexpected error ({class_name or 'unknown'}). Please try again."
