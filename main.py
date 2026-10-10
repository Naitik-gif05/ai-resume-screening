"""AI Resume Screening Pipeline.

Core pipeline (importable, no printing, no Streamlit dependency):
    extract_text()    -> raw text from a PDF/DOCX path or uploaded file object
    parse_job()       -> JobD
    parse_resume()    -> Resume
    final_score()     -> MatchResult
    screen_resumes()  -> (ranked_results, failures)

CLI:
    python main.py
"""

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Callable, Iterable

from docx import Document
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = "openai/gpt-oss-120b"
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
RATE_LIMIT_DELAY_SECONDS = 5  # pause after each LLM call (unchanged behaviour)

ResumeSource = Path | str | BinaryIO  # a path, or an uploaded file-like object

# on_progress(done, total, current_file, succeeded, failed)
ProgressCallback = Callable[[int, int, str | None, int, int], None]


@lru_cache(maxsize=1)
def get_client() -> Groq:
    """Create the Groq client on first use (not at import time)."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing")
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ResumeExtractionError(Exception):
    """A resume file could not be read or contained no text."""


class UnsupportedFileTypeError(ResumeExtractionError):
    """The file extension is not one of SUPPORTED_EXTENSIONS."""


class LLMOutputError(Exception):
    """The model returned invalid JSON or JSON that fails validation."""


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class JobD(BaseModel):
    role: str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float | None
    education_requirements: list[str]
    responsibilities: list[str]


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    score: float
    details: dict


jobd_schema = JobD.model_json_schema()
resume_schema = Resume.model_json_schema()
match_schema = MatchResult.model_json_schema()


# --------------------------------------------------------------------------
# File extraction (no LLM calls here)
# --------------------------------------------------------------------------


def _source_name(source: ResumeSource) -> str:
    """File name for a path or an uploaded file object (e.g. Streamlit)."""
    return Path(getattr(source, "name", str(source))).name


def _read_pdf(source: ResumeSource) -> str:
    reader = PdfReader(source)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def _read_docx(source: ResumeSource) -> str:
    document = Document(source)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text(source: ResumeSource) -> str:
    """Extract text from a PDF or DOCX.

    `source` may be a path or a file-like object with a `.name`
    (such as a Streamlit UploadedFile).

    Raises UnsupportedFileTypeError for other formats and
    ResumeExtractionError if the file is unreadable or has no text.
    """
    name = _source_name(source)
    suffix = Path(name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or name}' for {name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if hasattr(source, "seek"):
        source.seek(0)

    try:
        text = _read_pdf(source) if suffix == ".pdf" else _read_docx(source)
    except Exception as exc:
        raise ResumeExtractionError(f"Could not read {name}: {exc}") from exc

    if not text.strip():
        raise ResumeExtractionError(
            f"No text found in {name} (empty file, or a scanned/image-only document)"
        )

    return text


def find_resume_files(folder: Path) -> list[Path]:
    """Return supported resume files in a folder (case-insensitive extension)."""
    return sorted(
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# --------------------------------------------------------------------------
# LLM calls
# --------------------------------------------------------------------------


def _request_json(messages: list[dict], model_class: type[BaseModel], label: str):
    """Send messages to Groq in JSON mode and validate into `model_class`."""
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )

    raw_output = response.choices[0].message.content

    try:
        data = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMOutputError(f"{label}: model returned invalid JSON") from exc

    try:
        return model_class.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError(f"{label}: JSON failed validation: {exc}") from exc


def parse_job(job_description_text: str) -> JobD:
    """Parse a job description into a structured JobD."""
    system_prompt = f"""
You are an expert HR assistant.

Your job is to analyze a job description and extract
structured information from it.

Return ONLY valid JSON matching this schema:

{jobd_schema}

Important rules:
- Do not return the schema itself.
- Fill the schema with actual information from the job description.
- Do not invent information.
- If minimum experience is not mentioned, return null.
- If information for a list is missing, return an empty list.
"""

    user_prompt = f"""
Analyze the following job description:

{job_description_text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return _request_json(messages, JobD, "parse_job")


def parse_resume(resume_text: str) -> Resume:
    """Parse resume text into a structured Resume."""
    system_prompt = f"""
You are an expert resume parser.

Extract information from the resume based on its meaning,
not only based on exact section headings.

Different resumes may use different headings.

For example:
- Experience
- Professional Experience
- Work History
- Employment
- Internships

These may all contain relevant experience.

Skills may also appear in the skills section, work experience,
internships or projects.

Return ONLY valid JSON matching this schema:

{resume_schema}

Important rules:

1. Do not invent information.
2. If a value is not available, return null.
3. If a list has no information, return an empty list.
4. Include internships inside experiences.
5. Extract skills mentioned across the entire resume.
"""

    user_prompt = f"""
Parse the following resume:

{resume_text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return _request_json(messages, Resume, "parse_resume")


def final_score(job: JobD, resume: Resume) -> MatchResult:
    """Compare a parsed resume against a parsed job and score the match."""
    prompt = f"""
You are an HR recruiter.

Compare the candidate's resume with the job description.

JOB DESCRIPTION:
{job.model_dump_json(indent=2)}

CANDIDATE RESUME:
{resume.model_dump_json(indent=2)}

Return JSON matching this schema:

{match_schema}

Give me:

1. Candidate name
2. Matching skills
3. Missing important skills
4. Whether experience requirement is met
5. Overall match percentage from 0 to 100
6. A short final verdict

Keep the response concise and easy to read.
"""

    messages = [{"role": "user", "content": prompt}]

    return _request_json(messages, MatchResult, "final_score")


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def screen_resumes(
    job: JobD,
    resume_files: Iterable[ResumeSource],
    delay_seconds: float = RATE_LIMIT_DELAY_SECONDS,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[dict], list[dict]]:
    """Score every resume against `job`.

    `resume_files` can hold paths or uploaded file-like objects.

    Returns (ranked_results, failures):
      ranked_results: [{"file", "name", "score", "details", "resume"}, ...], best first
                      ("resume" is the parsed Resume as a dict)
      failures:       [{"file", "error"}, ...] for resumes that could not be processed

    One failing resume never stops the others. An empty input returns ([], []).

    `on_progress(done, total, current_file, succeeded, failed)` is called before
    each file starts and once more (current_file=None) when everything is done.
    """
    resume_files = list(resume_files)
    total = len(resume_files)
    all_results: list[dict] = []
    failures: list[dict] = []

    for index, resume_file in enumerate(resume_files):
        file_name = _source_name(resume_file)
        if on_progress:
            on_progress(index, total, file_name, len(all_results), len(failures))
        logger.info("\nProcessing: %s", file_name)

        try:
            resume_text = extract_text(resume_file)

            parsed_resume = parse_resume(resume_text)  # LLM call 1
            time.sleep(delay_seconds)

            result = final_score(job, parsed_resume)  # LLM call 2
            time.sleep(delay_seconds)
        except Exception as exc:  # recorded and reported, not swallowed
            error = f"{type(exc).__name__}: {exc}"
            # Log only the exception type: messages can contain resume content.
            logger.error("FAILED %s -> %s", file_name, type(exc).__name__)
            failures.append({"file": file_name, "error": error})
            continue

        logger.info("Score: %s", result.score)

        all_results.append(
            {
                "file": file_name,
                "name": parsed_resume.name,
                "score": result.score,
                "details": result.details,
                "resume": parsed_resume.model_dump(),
            }
        )

    if on_progress:
        on_progress(total, total, None, len(all_results), len(failures))

    all_results.sort(key=lambda candidate: candidate["score"], reverse=True)

    return all_results, failures


# --------------------------------------------------------------------------
# CLI display + entry point
# --------------------------------------------------------------------------

SAMPLE_JOB_DESCRIPTION = """
Software Development Engineer

Basic Qualifications:
- Experience with Python, Java, C++, or Go.
- Experience with data structures and algorithms.
- Bachelor's degree in Computer Science or related field.

Preferred Qualifications:
- Experience with AWS or other cloud platforms.
- Experience with SQL or NoSQL databases.
- Experience with Git and version control.
- Experience building software projects.

Responsibilities:
- Design and develop scalable software applications.
- Write clean and maintainable code.
- Participate in code reviews.
- Debug and troubleshoot software issues.
- Work with other engineers to build reliable systems.
"""


def display_results(results: list[dict], failures: list[dict]) -> None:
    """Print top 2 / lowest 2 candidates and any failed files."""
    top_2 = results[:2]
    # Avoid repeating the top candidates when there are fewer than 4 results.
    worst_2 = results[max(2, len(results) - 2):]

    print("\nTOP 2 CANDIDATES")
    for candidate in top_2:
        print(candidate["name"] or candidate["file"], "-", candidate["score"], "%")
        print(candidate["details"])

    if worst_2:
        print("\nLOWEST 2 CANDIDATES")
        for candidate in worst_2:
            print(candidate["name"] or candidate["file"], "-", candidate["score"], "%")
            print(candidate["details"])

    if failures:
        print("\nFAILED FILES")
        for failure in failures:
            print(failure["file"], "-", failure["error"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        job = parse_job(SAMPLE_JOB_DESCRIPTION)
    except Exception as exc:
        print(f"Could not parse the job description: {type(exc).__name__}: {exc}")
        return 1

    resume_folder = Path("resumes")
    if not resume_folder.is_dir():
        print(f"Resume folder not found: {resume_folder.resolve()}")
        return 1

    resume_files = find_resume_files(resume_folder)

    print(resume_files)
    print(job)

    if not resume_files:
        print("No PDF or DOCX resumes found in the 'resumes' folder.")
        return 0

    results, failures = screen_resumes(job, resume_files)
    display_results(results, failures)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
