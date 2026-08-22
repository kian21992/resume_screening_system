from app.services.education_domain import validate_experience_record
from app.services.nlp_pipeline import extract_education, extract_experience_records
from app.services.recommender import extract_resume_skills, estimate_decision_confidence


def test_pdf_bullets_and_wrapped_skill_lines_are_reconstructed():
    text = """SKILLS
● Knowledgeable  in MS Word, Excel, and Power
point
● Strong  dedication  and patience  in handling
children
● Knowledge  of theories  and practices  in handling  child's
development
● Strong  leadership  and managerial  skills
● Strong  interpersonal  skills
PERSONAL INFORMATION
"""

    assert extract_resume_skills(text) == [
        "Knowledgeable in MS Word, Excel, and Power point",
        "Strong dedication and patience in handling children",
        "Knowledge of theories and practices in handling child's development",
        "Strong leadership and managerial skills",
        "Strong interpersonal skills",
    ]


def test_cid_bullets_do_not_become_skills_or_split_comma_phrase():
    text = """SKILLS:
(cid:0)
Good communication and Writing Skills
(cid:0)
Video, photo, content editing skills
(cid:0)
Time management skills
WORK EXPERIENCE:
"""

    assert extract_resume_skills(text) == [
        "Good communication and Writing Skills",
        "Video, photo, content editing skills",
        "Time management skills",
    ]


def test_specific_degrees_survive_generic_level_headings():
    text = """EDUCATIONAL BACKGROUND
POST-GRADUATE
Master of Arts in Education Major in Educational Management
Don Honorio Ventura State University
August 2020 - September 2022
36 units
TERTIARY
Bachelor of Elementary Education Major in General Education (Cum Laude)
Mabalacat City College
11 May 2017
SECONDARY
REPUBLIC  CENTRAL  COLLEGES
1997 - March 2001
ELEMENTARY
DR. CLEMENTE N. DAYRIT SR. ELEM. SCHOOL
June 1991 - March 1997
SKILLS
"""

    records = extract_education(text)
    assert records[0]["degree"].startswith("Master of Arts in Education")
    assert records[0]["institution"] == "Don Honorio Ventura State University"
    assert records[1] == {
        "degree": "Bachelor of Elementary Education Major in General Education (Cum Laude)",
        "institution": "Mabalacat City College",
    }
    assert {record["institution"] for record in records} >= {
        "REPUBLIC CENTRAL COLLEGES",
        "DR. CLEMENTE N. DAYRIT SR. ELEM. SCHOOL",
    }


def test_school_then_degree_or_strand_rows_are_not_merged():
    text = """EDUCATION:
Tertiary School
Polytechnic University of the Philippines- Bachelor of Elementary Education
2021-2025
Secondary School
Bondoc Peninsula Agricultural High School- Information and Communication Technology Strand
2014-2020
Primary School
Northville 15 Integrated School
2008-2014
SKILLS:
"""

    assert extract_education(text) == [
        {
            "degree": "Bachelor of Elementary Education",
            "institution": "Polytechnic University of the Philippines",
        },
        {
            "degree": "Secondary School",
            "institution": "Bondoc Peninsula Agricultural High School",
        },
        {
            "degree": "Primary School",
            "institution": "Northville 15 Integrated School",
        },
    ]


def test_date_first_role_and_employer_rows_preserve_complete_history():
    text = """WORK EXPERIENCE
August 2, 2022 TO PRESENT
Principal, Nazarene Academy Inc.
Plan daily school activities.
June 30, 2021 TO PRESENT
OIC-Principal, Nazarene Academy Inc.
June 10, 2018- March 30, 2020
Elementary/Shadow Teacher, Bridges School of Angeles City
December 2017- March 31, 2018
Star Program Teacher, Westfields International School
EDUCATIONAL BACKGROUND
"""

    records = extract_experience_records(text)
    assert [(record["job_title"], record["company"]) for record in records] == [
        ("Principal", "Nazarene Academy Inc"),
        ("OIC-Principal", "Nazarene Academy Inc"),
        ("Elementary/Shadow Teacher", "Bridges School of Angeles City"),
        ("Star Program Teacher", "Westfields International School"),
    ]


def test_explicit_undated_work_is_kept_without_inventing_years():
    text = """WORK EXPERIENCE:
DSWD Tara Basa Tutors
Tutor
Delivered structured literacy support to learners.
Teaching Internship
Completed practice teaching at Mulanay Central Elementary School.
Freelance Tutor
Tutor
Provided one-on-one tutoring sessions.
CHARACTER REFERENCES:
"""

    records = extract_experience_records(text)
    assert [(record["job_title"], record["company"]) for record in records] == [
        ("Tutor", "DSWD Tara Basa Tutors"),
        ("Teaching Internship", "Mulanay Central Elementary School"),
        ("Freelance Tutor", "Not Identified"),
    ]
    assert all(record["years"] == 0 for record in records)
    assert all(record["duration_unknown"] for record in records)
    assert all(validate_experience_record(record)[0] for record in records)
    assert not validate_experience_record({
        "job_title": "Tutor",
        "company": "Example School",
        "years": 0,
    })[0]


def test_undated_work_cannot_produce_high_decision_confidence():
    level, reason = estimate_decision_confidence(
        recommendation_label="Not Qualified",
        resume_text=("Readable teaching resume with structured evidence. " * 60),
        contact_info={"name": "Example Candidate", "email": "candidate@example.com"},
        extracted_edu=[{"degree": "Bachelor of Education", "institution": "Example University"}],
        extracted_exp=[{
            "job_title": "Tutor",
            "company": "Example School",
            "years": 0,
            "duration_unknown": True,
        }],
        total_exp_years=0,
        required_skills=["Lesson Planning", "Classroom Management"],
        matched_skills=["Lesson Planning"],
        missing_skills=["Classroom Management"],
        matched_critical_skills=[],
        missing_critical_skills=[],
        skill_score=50,
        exp_score=0,
        edu_score=100,
        fit_score=43,
        experience_req=2,
        education_req="Bachelor's degree in Education",
    )

    assert level != "High"
    assert "dates were not provided" in reason


def test_labelled_education_rows_preserve_every_school_without_inventing_degree():
    text = """EDUCATIONAL BACKGROUND
College: La Concepcion College (2022-2023)
Senior High School: Badajoz Tablas College (2018-2019)
Junior High School: Eduardo M. Moreno National High School (2016-2017)
Elementary: Pang-alaalang Paaralang Severina M. Solidum (2012-2013)
SKILLS
"""

    assert extract_education(text) == [
        {
            "degree": "College Education",
            "institution": "La Concepcion College",
        },
        {
            "degree": "Senior High School",
            "institution": "Badajoz Tablas College",
        },
        {
            "degree": "Junior High School",
            "institution": "Eduardo M. Moreno National High School",
        },
        {
            "degree": "Elementary School",
            "institution": "Pang-alaalang Paaralang Severina M. Solidum",
        },
    ]


def test_bulleted_role_company_location_date_rows_preserve_complete_history():
    text = """WORK EXPERIENCE
- Private School Teacher
- La Concepcion College
- City of San Jose Del Monte, Bulacan
- 2023-2024
- Private School Teacher
- Nazarene Academy Inc.
- Angeles City, Pampanga
- 2025 - Present
PERSONAL INFORMATION
"""

    records = extract_experience_records(text)

    assert [
        (record["job_title"], record["company"], record["location"])
        for record in records
    ] == [
        (
            "Private School Teacher",
            "La Concepcion College",
            "City of San Jose Del Monte, Bulacan",
        ),
        (
            "Private School Teacher",
            "Nazarene Academy Inc",
            "Angeles City, Pampanga",
        ),
    ]
