"""Unit tests for ui_helpers (no Streamlit, no API calls).

Run from the project root:  python -m unittest discover -s tests -v
"""
import csv
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ui_helpers as h


class UploadTests(unittest.TestCase):
    def test_accepts_valid_pdf_and_docx(self):
        self.assertIsNone(h.validate_upload("a.pdf", b"%PDF-1.7 data"))
        self.assertIsNone(h.validate_upload("A.DOCX", b"PK\x03\x04data"))

    def test_rejects_bad_uploads(self):
        self.assertIn("unsupported", h.validate_upload("a.txt", b"x"))
        self.assertIn("empty", h.validate_upload("a.pdf", b""))
        self.assertIn("limit", h.validate_upload("a.pdf", b"%PDF" + b"0" * h.MAX_FILE_BYTES))
        self.assertIn("valid PDF", h.validate_upload("a.pdf", b"not a pdf"))
        self.assertIn("valid DOCX", h.validate_upload("a.docx", b"not a zip"))

    def test_unique_name(self):
        self.assertEqual(h.unique_name("cv.pdf", set()), "cv.pdf")
        self.assertEqual(h.unique_name("cv.pdf", {"cv.pdf"}), "cv (2).pdf")
        self.assertEqual(h.unique_name("cv.pdf", {"cv.pdf", "cv (2).pdf"}), "cv (3).pdf")

    def test_format_size(self):
        self.assertEqual(h.format_size(512), "512 B")
        self.assertEqual(h.format_size(2048), "2.0 KB")
        self.assertEqual(h.format_size(3 * 1024 * 1024), "3.0 MB")


class JobTextTests(unittest.TestCase):
    def test_description_only(self):
        self.assertEqual(h.compose_job_text("", "", " Build things ", "", "", 0), "Build things")

    def test_optional_fields_added(self):
        text = h.compose_job_text("SDE", "Acme", "Desc", "Python, SQL, python", "Go", 2.5)
        self.assertIn("Job title: SDE", text)
        self.assertIn("Company: Acme", text)
        self.assertIn("Required skills: Python, SQL", text)  # de-duplicated
        self.assertNotIn("python, SQL, python", text)
        self.assertIn("Preferred skills: Go", text)
        self.assertIn("Minimum experience: 2.5 years", text)
        self.assertTrue(text.endswith("Desc"))

    def test_estimate(self):
        self.assertEqual(h.estimate_minimum_seconds(3, 5), 30)
        self.assertEqual(h.estimate_minimum_seconds(0, 5), 0)


class ScoreTests(unittest.TestCase):
    def test_valid_score_boundaries(self):
        for value in (0, 100, 55.5, "70"):
            self.assertIsNotNone(h.valid_score(value), value)
        for value in (-0.1, 100.1, None, "abc", float("nan"), True):
            self.assertIsNone(h.valid_score(value), value)

    def test_ranking_order_and_invalid_scores(self):
        results = [
            {"file": "a", "score": 40, "details": {}},
            {"file": "b", "score": 95, "details": {}},
            {"file": "c", "score": 250, "details": {}},
            {"file": "d", "score": 70, "details": {}},
        ]
        ranked = h.normalize_results(results)
        self.assertEqual([r["file"] for r in ranked], ["b", "d", "a", "c"])
        self.assertEqual([r["rank"] for r in ranked], [1, 2, 3, None])
        metrics = h.summary_metrics(ranked, [{"file": "x"}])
        self.assertEqual(metrics["average"], (95 + 70 + 40) / 3)
        self.assertEqual(metrics["highest"], 95)
        self.assertEqual(metrics["invalid_scores"], 1)
        self.assertEqual(metrics["failed"], 1)
        self.assertEqual(metrics["total_attempted"], 5)

    def test_empty_metrics(self):
        metrics = h.summary_metrics([], [])
        self.assertIsNone(metrics["average"])
        self.assertIsNone(metrics["highest"])


class DetailsTests(unittest.TestCase):
    DETAILS = {
        "candidate_name": "Alice",
        "matching_skills": ["Python", "SQL"],
        "missing_important_skills": ["AWS"],
        "overall_match_percentage": 80,
        "final_verdict": "Strong fit.",
    }

    def test_extracts_only_what_exists(self):
        self.assertEqual(h.matching_skills(self.DETAILS), ["Python", "SQL"])
        self.assertEqual(h.missing_skills(self.DETAILS), ["AWS"])
        self.assertEqual(h.find_text(self.DETAILS), "Strong fit.")

    def test_missing_fields_return_none(self):
        self.assertIsNone(h.matching_skills({"foo": 1}))
        self.assertIsNone(h.missing_skills({}))
        self.assertIsNone(h.find_text({"foo": "bar"}))
        self.assertIsNone(h.matching_skills(None))

    def test_skill_check_is_case_insensitive_and_uses_experience_skills(self):
        resume = {"skills": ["python"], "experiences": [{"skills_used": ["SQL"]}]}
        found, unverified = h.skill_check(["Python", "sql", "Go"], resume)
        self.assertEqual(found, ["Python", "sql"])
        self.assertEqual(unverified, ["Go"])
        self.assertEqual(h.skill_check(["X"], {}), ([], ["X"]))


class ExportTests(unittest.TestCase):
    def test_csv_content_and_injection_guard(self):
        results = h.normalize_results([
            {"file": "a.pdf", "name": "=HYPERLINK(\"x\")", "score": 88,
             "details": DetailsTests.DETAILS, "resume": {"total_experience_years": 3}},
            {"file": "b.docx", "name": None, "score": 150, "details": {}, "resume": {}},
        ])
        rows = list(csv.reader(io.StringIO(h.results_to_csv(results))))
        self.assertEqual(rows[0], h.CSV_COLUMNS)
        first = dict(zip(rows[0], rows[1]))
        self.assertEqual(first["Rank"], "1")
        self.assertTrue(first["Candidate"].startswith("'="))
        self.assertEqual(first["Match score"], "88.0")
        self.assertEqual(first["Matching skills"], "Python; SQL")
        self.assertEqual(first["Missing or unverified skills"], "AWS")
        second = dict(zip(rows[0], rows[2]))
        self.assertEqual(second["Candidate"], "b.docx")  # falls back to file name
        self.assertEqual(second["Match score"], "")  # invalid score not exported as a number
        self.assertEqual(second["Rank"], "")

    def test_empty_export_has_header_only(self):
        rows = list(csv.reader(io.StringIO(h.results_to_csv([]))))
        self.assertEqual(rows, [h.CSV_COLUMNS])


class ErrorTests(unittest.TestCase):
    def test_messages(self):
        self.assertIn("not configured", h.friendly_error("ValueError: GROQ_API_KEY is missing"))
        self.assertIn("rejected", h.friendly_error("AuthenticationError: Error code: 401"))
        self.assertIn("rate limit", h.friendly_error("RateLimitError: 429 too many requests"))
        self.assertIn("network", h.friendly_error("APIConnectionError: Connection error."))
        self.assertIn("could not be read", h.friendly_error("LLMOutputError: parse_resume: JSON failed validation: secret@x.com"))
        self.assertEqual(
            h.friendly_error("ResumeExtractionError: No text found in a.pdf (empty file)"),
            "No text found in a.pdf (empty file)",
        )

    def test_unknown_errors_do_not_leak_text(self):
        message = h.friendly_error(RuntimeError("candidate john@example.com leaked"))
        self.assertNotIn("john", message)
        self.assertIn("RuntimeError", message)


if __name__ == "__main__":
    unittest.main()
