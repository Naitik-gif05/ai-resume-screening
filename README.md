# ResumeIQ — AI Resume Parser & Screening

ResumeIQ is an AI-assisted resume parsing and candidate screening application built with Python and Streamlit. It extracts information from PDF and DOCX resumes, compares candidate profiles with a job description using the Groq API, and presents match scores and explanations to support human review.

> **ResumeIQ is a decision-support tool, not an automated hiring system. All candidate information and AI-generated evaluations should be reviewed by a human.**

## Features

- Upload multiple PDF and DOCX resumes.
- Validate supported file formats and upload constraints.
- Extract candidate details such as skills, experience, education, projects, and certifications when available.
- Compare candidate profiles with a job description.
- Generate AI-based match scores and explanations.
- Rank candidates by match score.
- Review candidate details and screening results.
- Export screening results to CSV.
- Clear session data.
- Run automated tests for helper functionality.

## Technology Stack

- Python
- Streamlit — web interface
- Groq API — language-model processing
- Pydantic — structured data validation
- PyPDF — PDF text extraction
- python-docx — DOCX text extraction
- python-dotenv — environment-variable loading
- uv — dependency management
- pytest — automated testing

## Project Structure

```text
ResumeIQ/
├── app.py
├── main.py
├── ui_helpers.py
├── tests/
│   └── test_ui_helpers.py
├── .streamlit/
│   └── config.toml
├── resumes/              # Do not commit private candidate resumes
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

The `.env` file is used for local configuration and must not be committed to GitHub.

## Requirements

- Python version compatible with `pyproject.toml`
- `uv`
- A Groq API key

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Naitik-gif05/ai-resume-screening.git
cd ai-resume-screening
```

### 2. Install dependencies

Run this from the directory containing `pyproject.toml`:

```bash
uv sync
```

### 3. Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own key. Never commit `.env` or expose the key in source code, screenshots, or public repositories.

## Run the Application

From the project root:

```bash
uv run streamlit run app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`.

## How to Use

1. Open **Screen Resumes**.
2. Enter a job description and any optional role details.
3. Upload PDF or DOCX resumes.
4. Review the uploaded files and screening settings.
5. Start the analysis.
6. Review candidate scores, rankings, extracted details, and explanations.
7. Export results to CSV if needed.
8. Clear session data when finished.

## Run Tests

Run the helper test suite:

```bash
uv run python -m pytest tests/test_ui_helpers.py -v
```

Run all discovered tests:

```bash
uv run python -m pytest -v
```

## How It Works

1. Text is extracted from uploaded PDF and DOCX files.
2. The job description is processed into structured role information.
3. Resume text is processed to extract candidate information.
4. A language model compares each candidate profile with the job requirements and generates a match score and explanation.
5. Results are displayed and ranked for human review.

Match scores are AI-generated assessments. They are not probabilities of job performance or guarantees of candidate suitability.

## Limitations

- Model-generated information and scores may be incomplete or inaccurate.
- Scanned or image-only PDFs may not provide extractable text if OCR is unavailable.
- API rate limits and network issues may affect processing time.
- Large batches may take considerable time because requests are processed with pauses.
- The application has not been load-tested for enterprise-scale processing.
- Some skill comparisons may rely on text matching and may not recognize synonyms.
- The current implementation does not provide a validated, configurable weighted-scoring model.

## Security, Privacy, and Responsible Use

- Keep API keys, `.env`, private resumes, and candidate exports out of version control.
- Resume content is sent to the configured Groq API for processing. Review the provider's current data-handling terms before using personal or confidential information.
- Use sample data only when authorized.
- Verify extracted information against the original resume.
- Do not reject or select candidates solely on the basis of an AI-generated score. Human review is required.

## Future Improvements

Potential improvements include background processing for large batches, persistent storage, stronger retry and rate-limit handling, evidence-based explanations, OCR support, and additional export formats.

## License

No license is specified in this repository yet. Add a `LICENSE` file if you intend to distribute the project under an open-source license.
