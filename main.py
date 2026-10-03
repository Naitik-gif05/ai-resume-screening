from pypdf import PdfReader
from docx import Document
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
import time

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-120b"


job_description = """
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

class JobD(BaseModel):
    role: str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float | None
    education_requirements: list[str]
    responsibilities: list[str]

jobd_schema = JobD.model_json_schema()

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

{job_description}
"""

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]

response_format = {
    "type": "json_object"
}

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)

answer = response.choices[0].message.content


data = json.loads(answer)

job = JobD(**data)

resume_folder = Path("resumes")

resume_files = list(resume_folder.glob("*.pdf")) + list(resume_folder.glob("*.docx"))

print(resume_files)


print(job)


def extract_text(file_path):
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text

    elif file_path.suffix.lower() == ".docx":
        document = Document(file_path)

        text = ""

        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

        return text


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


resume_schema = Resume.model_json_schema()


def parse_resume(resume_text):
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
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"}
    )

    raw_output = response.choices[0].message.content

    data = json.loads(raw_output)

    resume = Resume(**data)
    return resume

class MatchResult(BaseModel):
    score: float
    details: dict

def final_score(job, resume):
    match_schema = MatchResult.model_json_schema()

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

    message = {
        "role": "user",
        "content": prompt
    }

    messages = [message]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)

    return MatchResult(**data)


all_results = []

for file_path in resume_files:
    print("\nProcessing:", file_path.name)

    resume_text = extract_text(file_path)

    parsed_resume = parse_resume(resume_text)
    # LLM call 1

    time.sleep(5)

    result = final_score(job, parsed_resume)
    # LLM call 2

    time.sleep(5)

    print("Score:", result.score)

    all_results.append({
        "name": parsed_resume.name,
        "score": result.score,
        "details": result.details
    })

all_results.sort(
    key=lambda candidate: candidate["score"],
    reverse=True
)

top_2 = all_results[:2]
worst_2 = all_results[-2:]

print("\nTOP 2 CANDIDATES")
for candidate in top_2:
    print(candidate["name"], "-", candidate["score"], "%")
    print(candidate["details"])

print("\nLOWEST 2 CANDIDATES")
for candidate in worst_2:
    print(candidate["name"], "-", candidate["score"], "%")
    print(candidate["details"])