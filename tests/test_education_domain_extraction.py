import json
from pathlib import Path

from app.services.education_domain import (
    classify_combined_heading,
    classify_section_heading,
    extract_education_entities,
    segment_resume,
    validate_education_record,
    validate_experience_record,
)
from app.services.nlp_pipeline import extract_education, extract_experience_records
from app.services.recommender import extract_resume_skills


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "education_resume_cases.json"


def test_section_classifier_handles_education_domain_and_combined_headings():
    assert classify_section_heading("TEACHING EXPERIENCE") == "experience"
    assert classify_section_heading("EDUCATIONAL BACKGROUND") == "education"
    assert classify_section_heading("COMPÉTENCES TECHNIQUES") == "skills"
    assert classify_combined_heading("CERTIFICATIONS, SKILLS & AWARDS") == [
        "certifications", "skills", "awards"
    ]


def test_segmenter_preserves_ordered_section_context():
    sections = segment_resume("MARIA\nSKILLS\nLesson Planning\nEDUCATION\nBachelor of Education")
    assert [section["section"] for section in sections] == ["header", "skills", "education"]
    assert sections[1]["lines"] == ["Lesson Planning"]


def test_hybrid_education_entities_include_evidence_and_confidence():
    entities = extract_education_entities(
        "TEACHING EXPERIENCE\nGrade 8 English Teacher\nUsed differentiated instruction.\n"
        "CERTIFICATIONS\nLicensed Professional Teacher"
    )
    entity_types = {entity["type"] for entity in entities}
    assert {"education_role", "grade_level", "subject", "education_skill", "license"}.issubset(entity_types)
    assert all(entity["evidence"] and entity["confidence"] in {"high", "medium"} for entity in entities)


def test_structured_validators_reject_noise_but_allow_unknown_employer():
    assert validate_experience_record({
        "job_title": "English Teacher", "company": "Not Identified", "years": 2
    })[0]
    assert not validate_experience_record({
        "job_title": "Prepared lessons for Grade 8 students",
        "company": "(A.Y", "years": 1,
    })[0]
    assert validate_education_record({
        "degree": "Bachelor of Secondary Education",
        "institution": "Example University",
    })[0]
    assert not validate_education_record({
        "degree": "High School With High Honors",
        "institution": "Unknown Institution",
    })[0]


def test_regression_resume_corpus():
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in cases:
        skills = extract_resume_skills(case["text"])
        experience = extract_experience_records(case["text"])
        education = extract_education(case["text"])

        for expected in case["expected_skills"]:
            assert expected in skills, (case["id"], expected, skills)
        assert [record["job_title"] for record in experience] == case["expected_roles"], case["id"]
        assert len(education) == case["expected_education_count"], case["id"]

