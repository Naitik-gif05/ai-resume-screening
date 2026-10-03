# AI Resume Screening & Evaluation System

An AI-powered resume screening system that evaluates multiple resumes against a dynamically generated Job Description (JD).

The project uses an LLM to extract structured information from both the job description and resumes, evaluate each candidate against the requirements, and rank candidates based on their overall match score.

## Features

- Generate a structured Job Description using an LLM
- Extract text from PDF and DOCX resumes
- Parse unstructured resumes into structured data
- Extract candidate information, skills, experience, education, projects, and certifications
- Compare candidates against job requirements
- Generate an overall match percentage
- Identify matching and missing skills
- Check experience requirements
- Rank candidates automatically
- Display the top 2 and lowest 2 candidates

## Architecture

```text
Job Description
       ↓
      LLM
       ↓
Structured JobD
       ↓
PDF / DOCX Resumes
       ↓
Text Extraction
       ↓
Raw Resume Text
       ↓
      LLM
       ↓
Structured Resume
       ↓
JobD + Resume
       ↓
      LLM
       ↓
Match Result
       ↓
Candidate Ranking
       ↓
Top 2 + Lowest 2
```

## Tech Stack

- Python
- Groq API
- `openai/gpt-oss-120b`
- Pydantic
- pypdf
- python-docx
- python-dotenv
- uv
- JSON

## Project Structure

```text
day5_project/
│
├── main.py
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
│
└── resumes/
    ├── Abhay Singh 1.pdf
    ├── Ashish Raj 24PCS007 (1) - Ashish Raj.pdf
    ├── Resume for Freshers - Priyanshu Singh.docx
    └── anshit verma resume word - Anshit Verma.docx
```

## How It Works

### 1. Job Description Generation

The system processes the job requirements using the LLM and converts them into a structured `JobD` Pydantic model.

The structured job data contains:

- Role
- Required skills
- Preferred skills
- Minimum experience
- Education requirements
- Responsibilities

### 2. Resume Text Extraction

The system supports both PDF and DOCX resumes.

- PDF → extracted using `pypdf`
- DOCX → extracted using `python-docx`

The extracted text is then passed to the LLM.

### 3. Resume Parsing

The LLM converts the raw resume text into a structured `Resume` model containing:

- Name
- Email
- Phone
- Total experience
- Skills
- Experiences
- Education
- Projects
- Certifications

### 4. Candidate Evaluation

Each structured resume is compared with the structured Job Description.

The LLM evaluates:

- Matching skills
- Missing important skills
- Experience requirement
- Overall candidate-job match
- Final verdict

The result is stored as a structured `MatchResult`.

### 5. Candidate Ranking

All candidates are collected and sorted according to their match score.

The system then displays the top 2 and lowest 2 candidates.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd day5_project
```

Install the project dependencies:

```bash
uv sync
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

The API key is loaded using `python-dotenv`.

**Never commit your `.env` file or API keys to GitHub.**

## Running the Project

Run the application using:

```bash
uv run python main.py
```

The system will process the resumes present inside the `resumes/` directory and generate candidate scores and rankings.

## Example Output

```text
Processing: Abhay Singh 1.pdf
Score: 60.0

Processing: Ashish Raj 24PCS007 (1) - Ashish Raj.pdf
Score: 70.0

Processing: anshit verma resume word - Anshit Verma.docx
Score: 40.0

Processing: Resume for Freshers - Priyanshu Singh.docx
Score: 30.0


TOP 2 CANDIDATES

Ashish Raj - 70.0%
Abhay Pratap Singh - 60.0%


LOWEST 2 CANDIDATES

Anshit Verma - 40.0%
Priyanshu Singh - 30.0%
```

> Match scores are generated using an LLM and may vary between runs.

## Key Concepts Learned

- LLM-based information extraction
- Structured outputs
- Pydantic models
- JSON parsing
- PDF and DOCX processing
- Prompt engineering
- Multi-document processing
- LLM-based candidate evaluation
- Candidate ranking
- Environment variable management
- Dependency management with uv

## Learning Context

This project was built as part of the Padho with Pratyush AI Engineer learning track, focusing on practical applications of LLMs, structured outputs, and document processing.

## Future Improvements

- Add a web interface
- Add resume and job-description upload functionality
- Add batch processing for larger resume datasets
- Add deterministic scoring alongside LLM evaluation
- Add evaluation metrics for screening accuracy
- Add database storage for candidates
- Add observability and evaluation pipelines