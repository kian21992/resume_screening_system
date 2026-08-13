"""
Extraction accuracy harness (ground-truth evaluation).

WHAT THIS IS FOR
    Measures how accurately the NLP pipeline extracts fields from resumes by
    comparing its output against hand-labeled "correct" answers. It reports
    precision / recall / F1 per field — the exact metrics the thesis Data
    Analysis section promises — so extraction changes become measurable instead
    of guesswork, and the printed table can go straight into the paper.

HOW TO USE IT WITH YOUR REAL RESUMES (recommended)
    1. Put real resume files in a folder (PDF/DOCX).
    2. For each, add an entry to GROUND_TRUTH below: the file path plus the
       correct values you expect (email, degree_rank, years_experience within a
       tolerance, and the set of skills that are genuinely present).
    3. Run:  python -m tests.extraction_ground_truth
    The SAMPLES below are synthetic examples so the harness runs out of the box;
    replace/extend them with your labeled real resumes.

FIELDS SCORED
    - email            exact match
    - degree_rank      exact match (0-5; see recommender.get_degree_rank)
    - years_experience counted correct if within +/- YEARS_TOLERANCE
    - skills           precision/recall/F1 over the set of expected skills
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.nlp_pipeline import (
    extract_contact_info,
    extract_years_of_experience,
    extract_education,
)
from app.services.recommender import get_degree_rank, analyze_skills

YEARS_TOLERANCE = 1.0  # years-of-experience counted correct if within this band


# ---------------------------------------------------------------------------
# Ground-truth entries.
# Use "text" for inline resume text, OR "path" to load a real PDF/DOCX file
# (extract_text_from_file will be used automatically when "path" is present).
# ---------------------------------------------------------------------------
GROUND_TRUTH = [
    {
        "id": "teacher_paraphrased",
        "text": (
            "Ana Cruz | ana@example.com | 0917-555-2020\n"
            "Teaching Experience\n"
            "English Teacher, San Jose High School   June 2016 - June 2024\n"
            "Planned daily lessons and managed a classroom of 40 students.\n"
            "Education\n"
            "Bachelor of Secondary Education major in English, "
            "Holy Angel University 2012 - 2016\n"
        ),
        "email": "ana@example.com",
        "degree_rank": 3,             # Bachelor's
        "years_experience": 8.0,
        "skills_to_check": ["Lesson Planning", "Classroom Management"],
        "skills_expected_present": ["Lesson Planning", "Classroom Management"],
    },
    {
        "id": "engineer_no_heading",
        "text": (
            "Juan Dela Cruz\njuan@mail.com\n+63 917 555 1234\n"
            "Software Engineer, Acme Inc   Jan 2018 - Jan 2023\n"
            "Built REST APIs in Python and SQL.\n"
            "Bachelor of Science in Computer Science, State University 2013 - 2017\n"
        ),
        "email": "juan@mail.com",
        "degree_rank": 3,
        "years_experience": 5.0,
        "skills_to_check": ["Python", "SQL", "Java"],
        "skills_expected_present": ["Python", "SQL"],
    },
]


def _resume_text(entry):
    if "path" in entry and entry["path"]:
        from app.services.extractor import extract_text_from_file
        return extract_text_from_file(entry["path"])
    return entry.get("text", "")


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def run():
    email_hits = years_hits = degree_hits = total = 0
    skill_tp = skill_fp = skill_fn = 0

    print(f"{'resume':<26}{'email':<7}{'degree':<8}{'years':<8}{'skills(matched/expected)'}")
    print("-" * 72)

    for entry in GROUND_TRUTH:
        text = _resume_text(entry)
        total += 1

        got_email = extract_contact_info(text).get("email")
        email_ok = (got_email == entry["email"])
        email_hits += email_ok

        extracted_edu = extract_education(text)
        cand_rank = (max(get_degree_rank(e["degree"]) for e in extracted_edu)
                     if extracted_edu else get_degree_rank(text))
        degree_ok = (cand_rank == entry["degree_rank"])
        degree_hits += degree_ok

        got_years = extract_years_of_experience(text)
        years_ok = abs(got_years - entry["years_experience"]) <= YEARS_TOLERANCE
        years_hits += years_ok

        matched, _ = analyze_skills(text, entry["skills_to_check"])
        expected = set(entry["skills_expected_present"])
        matched_set = set(matched)
        skill_tp += len(matched_set & expected)
        skill_fp += len(matched_set - expected)
        skill_fn += len(expected - matched_set)

        print(f"{entry['id']:<26}"
              f"{'OK' if email_ok else 'X':<7}"
              f"{('OK' if degree_ok else f'X({cand_rank})'):<8}"
              f"{('OK' if years_ok else f'X({got_years})'):<8}"
              f"{len(matched_set & expected)}/{len(expected)}")

    print("-" * 72)
    n = max(total, 1)
    print(f"email   accuracy: {email_hits}/{total} = {email_hits / n:.0%}")
    print(f"degree  accuracy: {degree_hits}/{total} = {degree_hits / n:.0%}")
    print(f"years   accuracy: {years_hits}/{total} = {years_hits / n:.0%} "
          f"(within +/-{YEARS_TOLERANCE}y)")
    p, r, f1 = _prf(skill_tp, skill_fp, skill_fn)
    print(f"skills  precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}")


if __name__ == "__main__":
    run()