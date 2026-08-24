from app.services.education_domain import validate_experience_record
from app.services.extractor import (
    _clean_extracted_text,
    _select_best_pdf_text,
    _text_quality_score,
)
from app.services.nlp_pipeline import (
    extract_contact_info,
    extract_certifications,
    extract_education,
    extract_experience_records,
)
from app.services.recommender import (
    evaluate_candidate,
    extract_resume_skills,
    estimate_decision_confidence,
)


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


def test_other_educational_achievements_heading_stops_skill_extraction():
    text = """SKILLS
Computer Literate (Microsoft Software
Application)
Effective Written Communication
Oral Communication Skills
Classroom Management
Critical Thinking Skills
Reliable
OTHER EDUCATIONAL ACHIEVEMENTS AND EXPERIENCES
Licensed Professional Teacher Board Passer
Academic Excellence Award
CHARACTER REFERENCES
"""

    assert extract_resume_skills(text) == [
        "Computer Literate (Microsoft Software Application)",
        "Effective Written Communication",
        "Oral Communication Skills",
        "Classroom Management",
        "Critical Thinking Skills",
        "Reliable",
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


def test_pdf_selector_rejects_work_heading_orphaned_after_its_entries():
    correctly_ordered = """JOYCE ANN R. PINEDA
WORK EXPERIENCE
Public Senior High School Teacher
August 29, 2023 - Present
Pampanga High School
Senior High School Teacher
January 13, 2021 - May 31, 2023
Colegio De Sebastian Pampanga Inc.
EDUCATION
Bachelor of Secondary Education
City College of San Fernando
SKILLS
Classroom Management
Lesson Planning
"""
    wrong_column_order = correctly_ordered.replace("WORK EXPERIENCE\n", "")
    wrong_column_order += "Additional sidebar contact detail\nWORK EXPERIENCE\n"

    assert _text_quality_score(correctly_ordered) > _text_quality_score(
        wrong_column_order
    )

    selected, method = _select_best_pdf_text([
        ("pdfplumber", correctly_ordered),
        ("PyPDF2", wrong_column_order),
    ])

    assert method == "pdfplumber"
    assert selected == correctly_ordered


def test_pdf_selector_rejects_word_per_line_fragmentation():
    readable = """MARLYN ROXAS CARREON, LPT.
Mexico, Pampanga
09169143899
marlyn@example.com
WORK OBJECTIVE:
To obtain a teaching position where I can nurture students and impart my skills and abilities.
EDUCATIONAL BACKGROUND:
Post-Graduate Master in Arts in Education
Pampanga State University
S.Y. 2025-present
Tertiary Bachelor of Secondary Education
Pampanga State University
S.Y. 2014-2018
WORK EXPERIENCES:
Nov. 2018 - Jan. 2019 ESL Teacher, NK International Academy
June 2019 - June 2022 Infant Jesus Academy, Pampanga
May 2023 - present Sucad National High School
SEMINARS AND TRAININGS:
2025 Division Training of Teacher-Tutors on Academic Recovery and Accessible Learning
2023 Seminar Workshop for Teachers in Junior and Senior High School
2022 Basic Computer Literacy
2021 Teaching and Assessing K-12 Standards Across Learning Modalities
2020 Whole Child Education Roadshow for Filipino Learners
2019 Curriculum Mapping Workshop
PERSONAL DATA:
Date of Birth June 9, 1997
Civil Status Single
Language Tagalog Kapampangan English
CHARACTER REFERENCES:
Nherie Carreon Head Manager JGT Corporation
Jayem Sanchez Senior High Teacher Betis National High School
I certify that all information shown is true and correct.
"""
    fragmented = "\n".join(readable.split())

    assert _text_quality_score(readable) > _text_quality_score(fragmented)

    selected, method = _select_best_pdf_text([
        ("pdfplumber", readable),
        ("PyPDF2", fragmented),
    ])

    assert method == "pdfplumber"
    assert selected == readable


def test_role_then_date_employer_and_location_preserves_complete_record():
    text = """WORK EXPERIENCE
Customer Service Associate
2020 - 2021 - Concentrix
Clark Freeport Zone, Pampanga, Philippines
EDUCATION
"""

    assert extract_experience_records(text) == [{
        "job_title": "Customer Service Associate",
        "company": "Concentrix",
        "location": "Clark Freeport Zone, Pampanga, Philippines",
        "years": 1.0,
    }]


def test_unrelated_degree_and_work_are_not_described_as_teacher_matches():
    text = """CHARLENE T. COCHON
charlene@example.com | 09937013345
EDUCATIONAL BACKGROUND
Bachelor of Science Business Administration Major in Marketing
Don Honorio Ventura Technological State University
High School
San Isidro High School
Elementary Education
San Isidro Elementary School
WORK EXPERIENCE
Auditor
Precious Loyal Pet
2023 - 2024
Sales and Marketing Assistant
NGT Venture Inc.
2021 - 2023
Cashier / Front Desk
Reliability Confidence & Gas Corporation
2019 - 2021
SKILLS
Microsoft Office
Marketing Promotion
Accounting
"""

    result = evaluate_candidate(
        resume_text=text,
        job_desc_text="Teacher Lesson Planning Classroom Management",
        required_skills=["Lesson Planning", "Classroom Management"],
        experience_req=2,
        education_req="Bachelor's degree in Education",
        job_title="TEACHER",
    )

    assert result["education_score"] == 50.0
    assert result["experience_score"] == 40.0
    assert "total year(s) of work history" in result["summary"]
    assert "limited relevance to the TEACHER role" in result["summary"]
    assert "degree field does not fully match" in result["summary"]
    assert "work experience advantage" not in result["decision_explanation"]
    assert "experience is far below the minimum requirement" not in result["decision_explanation"]
    assert "work history has limited relevance" in result["decision_explanation"]


def test_leading_colon_does_not_hide_role_or_create_reused_title_duplicate():
    text = """WORK EXPERIENCES
: Instructor II
Pampanga State University
2023- Present
Instructor I
Don Honorio Ventura State University
2020-2023
Teacher I
Department of Education
2019-2020
Class Adviser
Infant Jesus Academy
2015-2019
"""

    records = extract_experience_records(text)

    assert [(record["job_title"], record["company"]) for record in records] == [
        ("Instructor II", "Pampanga State University"),
        ("Instructor I", "Don Honorio Ventura State University"),
        ("Teacher I", "Department of Education"),
        ("Class Adviser", "Infant Jesus Academy"),
    ]


def test_level_first_education_and_wrapped_internship_titles_are_preserved():
    text = """EDUCATIONAL BACKGROUND
College
Bachelor of Secondary Education Major in Filipino
Pampanga State University (former Don Honorio
Ventura State University)
(2020-2024)
Senior High School
Humanities and Social Sciences Strand
Pulung Santol National High School
(2018-2020)
Junior High School
Pulung Santol National High School
(2014-2018)
Elementary
Pulung Santol Elementary School
(2008-2014)
WORK EXPERIENCE
Private Tutor
(April 2025-Present)
English as Secondary Language (ESL) Teacher
SAT English Center
(June 2024 - December 2024)
Student Internship handled Grade 7, Grade 9,
and Grade 10
Pulung Santol National High School
(September 2023 - May 2024)
Literary Training Services (Tutoring Kids)
Don Honorio Ventura State University
(2020-2021)
Work Immersion (Student Teacher)
Becuran National High School
(December 2019 - January 2020)
SKILLS
"""

    assert extract_education(text) == [
        {
            "degree": "Bachelor of Secondary Education Major in Filipino",
            "institution": (
                "Pampanga State University (former Don Honorio Ventura State University)"
            ),
        },
        {
            "degree": "Senior High School - Humanities and Social Sciences Strand",
            "institution": "Pulung Santol National High School",
        },
        {
            "degree": "Junior High School",
            "institution": "Pulung Santol National High School",
        },
        {
            "degree": "Elementary School",
            "institution": "Pulung Santol Elementary School",
        },
    ]

    records = extract_experience_records(text)
    assert [(record["job_title"], record["company"]) for record in records] == [
        ("English as Secondary Language (ESL) Teacher", "SAT English Center"),
        (
            "Student Internship handled Grade 7, Grade 9, and Grade 10",
            "Pulung Santol National High School",
        ),
        (
            "Literary Training Services (Tutoring Kids)",
            "Don Honorio Ventura State University",
        ),
        ("Work Immersion (Student Teacher)", "Becuran National High School"),
        ("Private Tutor", "Not Identified"),
    ]


def test_header_single_name_and_plural_bachelors_degree_are_supported():
    text = """Komala
Sr Business System Analyst/Scrum Master
PROFILE:
Experienced business analyst.
EDUCATION:
Bachelors of Technology in Electrical Engineering.
"""

    assert extract_contact_info(text)["name"] == "Komala"
    assert extract_education(text) == [{
        "degree": "Bachelors of Technology in Electrical Engineering",
        "institution": "Unknown Institution",
    }]


def test_applicant_signature_is_strong_name_evidence_and_dates_are_not_skills():
    text = """CURRICULUM VITAE
BAL LESTEROS , JEANNE PAULINE B . Brgy. Sto. Domingo, Capas, Tarlac
jeannepauline03@gmail.com
SKILLS
Computer Literate
NC holder CLASP (Communication, Language and Skills Program)
April 2016
ACHIEVEMENTS/AWARDS:
Ballesteros, Jeanne Pauline B.
Applicant
"""

    assert extract_contact_info(text)["name"] == "Jeanne Pauline B. Ballesteros"
    assert extract_resume_skills(text) == [
        "Computer Literate",
        "NC holder CLASP (Communication, Language and Skills Program)",
    ]


def test_template_labels_do_not_pollute_education_or_undated_work():
    text = """Jennifer O. Malang
EDUCATION
[Name of College/University] DON HONORIO VENTURA COLLEGE OF ARTS AND TRADES
Bachelor of [Degree] BACHELOR OF SCINCE IN INDUSTRIAL EDUCATION (2002-2006)
[Name of School] N/A
Senior High School
[Name of School] SAN ISIDRO HIGH SCHOOL (1998-2002)
Junior High School
EXPERIENCE
Kindergarten Teacher
[Magliman Integrated School]
Plan and implement learning activities.
TECHNICAL SKILLS
"""

    assert extract_education(text) == [
        {
            "degree": "BACHELOR OF SCIENCE IN INDUSTRIAL EDUCATION",
            "institution": "DON HONORIO VENTURA COLLEGE OF ARTS AND TRADES",
        },
        {
            "degree": "Senior High School",
            "institution": "Unknown Institution",
        },
        {
            "degree": "Junior High School",
            "institution": "SAN ISIDRO HIGH SCHOOL",
        },
    ]
    assert extract_experience_records(text) == [{
        "job_title": "Kindergarten Teacher",
        "company": "Magliman Integrated School",
        "location": "Not Identified",
        "years": 0.0,
        "duration_unknown": True,
    }]


def test_known_highest_degree_field_outweighs_generic_duplicate():
    text = """EDUCATION
Bachelor's of Arts
Bachelor of Arts in Political Science and Law
WORK EXPERIENCE
HR Personnel Assistant
Example Company
2020-2022
"""

    result = evaluate_candidate(
        resume_text=text,
        job_desc_text="Teacher Lesson Planning",
        required_skills=[],
        experience_req=0,
        education_req="Bachelor's degree in Education",
        job_title="Teacher",
    )

    assert result["education_score"] == 50.0


def test_duration_table_rows_preserve_each_employer_and_role():
    text = """PROFESSIONAL EXPERIENCE
KOHLS, Menomonee Falls, WI                                      DURATION
Sr. Business System Analyst/ Scrum Master                       AUG 2016 - PRESENT
PROJECT DESCRIPTION: Store order tracking.
NJDOC, Trenton, NJ                                              DURATION
Sr. Business System Analyst                                     MARCH 2015 - JULY 2016
PROJECT DESCRIPTION: Education data mart.
AMWAY, Ada, MI                                                  DURATION
Sr. Business Analyst                                            JAN 2014 - FEB 2015
PROJECT DESCRIPTION: Product data improvements.
Mutual Insurance, Mumbai, INDIA                                 DURATION
Business Analyst                                                JULY 2010 - MARCH 2011
PROJECT DESCRIPTION: Commercial insurance data mart.
OLX, Hyderabad, INDIA                                           DURATION
Jr. Business Analyst                                            OCT 2009 - JUNE 2010
EDUCATION
"""

    records = extract_experience_records(text)
    assert [(record["job_title"], record["company"]) for record in records] == [
        ("Sr. Business System Analyst/ Scrum Master", "KOHLS"),
        ("Sr. Business System Analyst", "NJDOC"),
        ("Sr. Business Analyst", "AMWAY"),
        ("Business Analyst", "Mutual Insurance"),
        ("Jr. Business Analyst", "OLX"),
    ]
    assert [record["location"] for record in records] == [
        "Menomonee Falls, WI",
        "Trenton, NJ",
        "Ada, MI",
        "Mumbai, INDIA",
        "Hyderabad, INDIA",
    ]


def test_singular_certification_heading_and_adjacent_skill_are_supported():
    text = """CERTIFICATION:
Scrum Master Accredited Certification (International Scrum Institute).
Lean Six Sigma Green Belt Certification
TECHNICAL SKILLS
Basic Computer Skills Parent-Teacher Communication
WORK EXPERIENCE
"""

    assert [
        record["certification_name"] for record in extract_certifications(text)
    ] == [
        "Scrum Master Accredited Certification (International Scrum Institute)",
        "Lean Six Sigma Green Belt Certification",
    ]
    assert extract_resume_skills(text) == [
        "Basic Computer Skills",
        "Parent-Teacher Communication",
    ]


def test_credential_section_rejects_issuer_heading_and_narrative_duplicates():
    text = """CERTIFICATIONS
Licensed Professional Teacher
Licensed Professional Teacher with experience in classroom instruction
Professional Regulation Commission (PRC)
SCHOOL DESIGNATIONS / LEADERSHIP ROLES
Grade Level Chairperson
WORK EXPERIENCE
"""

    assert extract_certifications(text) == [{
        "certification_name": "Licensed Professional Teacher",
        "credential_type": "Professional License",
        "issuer": None,
        "date_obtained": None,
    }]


def test_designations_heading_stops_training_credential_extraction():
    text = """SEMINARS AND TRAININGS
Classroom Assessment Workshop
SCHOOL DESIGNATIONS / LEADERSHIP ROLES
Grade Level Chairperson
WORK EXPERIENCE
"""

    names = [
        record["certification_name"] for record in extract_certifications(text)
    ]
    assert "SCHOOL DESIGNATIONS / LEADERSHIP ROLES" not in names
    assert "Grade Level Chairperson" not in names


def test_educational_attainment_is_contained_and_reopens_after_awards():
    text = """Maria Alpha S. Bersabal
WORK EXPERIENCE
Off-Campus Teaching
School of the Future
November 2014 - February 2015
Student Teacher handling elementary pupils
EDUCATIONAL ATTAINMENT
TERTIARY:
BACHELOR OF ELEMENTARY EDUCATION
Central Bicol State University of Agriculture
June 2011 - April 2015
SCHOLARSHIPS AND HONORS RECEIVED
Academic Scholar
SECONDARY:
Nabua National High School
June 2007 - April 2011
ELEMENTARY:
La Opinion Elementary School
June 2001 - April 2007
TRAININGS/SEMINARS ATTENDED
TEACHING STRATEGIES FOR ELEMENTARY LEARNERS
Nabua National High School
February 2015
CHARACTER REFERENCES
Program Chairman, Elementary Education Program
CBSUA College of Development Education
APPLICATION LETTER
I earned a Bachelor's degree in Elementary Education for the development of learners.
"""

    assert extract_education(text) == [
        {
            "degree": "BACHELOR OF ELEMENTARY EDUCATION",
            "institution": "Central Bicol State University of Agriculture",
        },
        {
            "degree": "High School",
            "institution": "Nabua National High School",
        },
        {
            "degree": "Elementary School",
            "institution": "La Opinion Elementary School",
        },
    ]
    assert [
        (record["job_title"], record["company"])
        for record in extract_experience_records(text)
    ] == [("Off-Campus Teaching", "School of the Future")]
    assert [
        record["certification_name"] for record in extract_certifications(text)
    ] == ["TEACHING STRATEGIES FOR ELEMENTARY LEARNERS"]


def test_professional_qualification_and_duties_do_not_create_false_jobs():
    text = """MR PHILIP O. PATIGDAS
Mesaieed Qatar
Mobile No. +974 66199309
PROFESSIONAL QUALIFICATION
Master's in Education (M.Ed.) Major in Administration & Supervision
Cebu Technological University
Completed Comprehensive Academic Requirements leading to M.Ed
Summer 2002
Bachelor in Elementary Education (BEED) Major in General Science
Cebu Normal University
Graduated Year 1997
WORK HISTORY
Classroom Teacher (Private School)
Marie Ernestine School
From 1997-2000
Advisory Teacher for Grade 5
Subject Teacher for Science
Medium of Instruction
English
Classroom Adviser (Public School)
Department of Education - Talima Elementary School
From 2000-2009
Teaching Science, English and Mathematics
Time spent for teaching
Private Tutor (Home Service)
Tutorial classes at home
From 2004-2007
Senior Receptionist/Administrative Assistant
AMWAJ
From 2009-present
Report directly to manager
Assist customers and monitor office functions
Tutorials (Home Service)
From June 2010-May 2011
Qatar
Teaching English and Mathematics
SEMINARS ATTENDED
ENHANCEMENT OF TEACHING STRATEGIES
May 2008
Lapu-Lapu City Central Elementary School
TECHNICAL SKILLS
Microsoft Word
"""

    assert extract_contact_info(text)["phone"] == "+974 66199309"
    assert extract_education(text) == [
        {
            "degree": (
                "Master's in Education (M.Ed.) Major in Administration & Supervision "
                "(Ongoing/Incomplete)"
            ),
            "institution": "Cebu Technological University",
        },
        {
            "degree": "Bachelor in Elementary Education (BEED) Major in General Science",
            "institution": "Cebu Normal University",
        },
    ]
    assert [
        (record["job_title"], record["company"])
        for record in extract_experience_records(text)
    ] == [
        ("Classroom Teacher (Private School)", "Marie Ernestine School"),
        (
            "Classroom Adviser (Public School)",
            "Department of Education - Talima Elementary School",
        ),
        ("Private Tutor (Home Service)", "Tutorial classes at home"),
        ("Senior Receptionist/Administrative Assistant", "AMWAJ"),
        ("Position Not Stated", "Tutorials (Home Service)"),
    ]
    assert [
        record["certification_name"] for record in extract_certifications(text)
    ] == ["ENHANCEMENT OF TEACHING STRATEGIES"]


def test_pdf_selector_rejects_orphaned_heading_run_before_resume_content():
    coordinate_order = """CHRISTINE JOY CASTRO ANGELES
Address: #48 Purok 3, Baliti City of San Fernando Pampanga
Phone: 0999-555-8584
Email: christinejoyangeles17@gmail.com
PROFILE
Financial Management graduate seeking an entry-level role.
WORK EXPERIENCE
Accounting Staff, Our Lady of Fatima University Inc. June 29, 2016 - Present
Maintain financial documents and prepare billing invoices.
TRAININGS AND SEMINARS
Know your Money November 2025
Customer Service Training 2026 Enhancing the Culture of Excellent Service
Customer Service Training 2026 Module II
Going the Extra Mile: Giving Your Best at Work and in Everyday Life
EDUCATION
Bachelor of Science Business and Administration 2012 - 2016
Major in Financial Management
Colegio de Sebastian
SKILLS
Exceptional communication skills
Successful working in a team environment, as well as independently.
Microsoft Office Suite
PERSONAL INFORMATION
Name:
Angeles, Christine Joy C.
Civil Status: Single
CHARACTER REFERENCES
"""
    content_stream_order = """TRAININGS AND SEMINARS
EDUCATION
WORK EXPERIENCE
PROFILE
Know your Money November 2025
2012 - 2016
Bachelor of Science Business and Administration
Major in Financial Management
Colegio de Sebastian
June 29, 2016 - Present
Accounting Staff, Our Lady of Fatima University Inc.Maintain financial documents.
Financial Management graduate seeking an entry-level role.
CHRISTINE JOY CASTRO ANGELES
Address:
Phone:
Email:
#48 Purok 3, Baliti City of San Fernando Pampanga
0999-555-8584
christinejoyangeles17@gmail.com
SKILLS
Exceptional communication skills
Microsoft Office Suite
PERSONAL INFORMATION
Civil Status: Single
CHARACTER REFERENCES
"""

    selected, method = _select_best_pdf_text([
        ("pdfplumber", coordinate_order),
        ("PyPDF2", content_stream_order),
    ])

    assert method == "pdfplumber"
    assert selected == coordinate_order
    assert extract_contact_info(selected) == {
        "name": "Christine Joy Castro Angeles",
        "email": "christinejoyangeles17@gmail.com",
        "phone": "0999-555-8584",
    }
    assert extract_education(selected) == [{
        "degree": (
            "Bachelor of Science Business and Administration, "
            "Major in Financial Management"
        ),
        "institution": "Colegio de Sebastian",
    }]
    assert [
        (record["job_title"], record["company"])
        for record in extract_experience_records(selected)
    ] == [("Accounting Staff", "Our Lady of Fatima University Inc")]
    assert extract_resume_skills(selected) == [
        "Exceptional communication skills",
        "Successful working in a team environment, as well as independently",
        "Microsoft Office Suite",
    ]
    assert [
        (record["certification_name"], record["date_obtained"])
        for record in extract_certifications(selected)
    ] == [
        ("Know your Money", "November 2025"),
        (
            "Customer Service Training Enhancing the Culture of Excellent Service",
            "2026",
        ),
        (
            "Customer Service Training Module II Going the Extra Mile: Giving Your Best "
            "at Work and in Everyday Life",
            "2026",
        ),
    ]


def test_wrapped_identity_credential_and_education_history_stay_in_their_sections():
    text = _clean_extracted_text("""LYNDO V.
FONTANILLA
L I C E N S E D  P R O F E S S I O N A L
T E A C H E R
CONTACT INFORMATION
glyndzfontanilla@gmail.com
EDUCATIONAL HISTORY
St. Anthony College and Technology
Masters of Art in Educational Management (candidate, 36 units earned)
Angeles University Foundation
Bachelor of Secondary Education major in Music, Art, Physical Education and
Health (BSEd-MAPEH)
Ranao National High School
Secondary Education
Macaboboni Elementary School
Primary Education
INTERESTS AND HOBBIES
PowerPoint presentation design
""")

    assert extract_contact_info(text)["name"] == "Lyndo V. Fontanilla"
    assert extract_certifications(text)[0]["certification_name"] == "Licensed Professional Teacher"
    assert [
        (record["degree"], record["institution"])
        for record in extract_education(text)
    ] == [
        (
            "Masters of Art in Educational Management (candidate, 36 units earned) (Ongoing/Incomplete)",
            "St. Anthony College and Technology",
        ),
        (
            "Bachelor of Secondary Education major in Music, Art, Physical Education and Health (BSEd-MAPEH)",
            "Angeles University Foundation",
        ),
        ("Secondary Education", "Ranao National High School"),
        ("Primary Education", "Macaboboni Elementary School"),
    ]


def test_parallel_language_column_does_not_split_or_pollute_skills():
    text = """QUALIFICATIONS AND SKILLS
LANGUAGE
- Skilled in creating and formatting documents,
- Tagalog
presentations, and spreadsheets using Microsoft Word,
- English
PowerPoint, and Excel.
- Self-motivated and well people oriented
PROFESSIONAL EXPERIENCE
"""

    assert extract_resume_skills(text) == [
        "Skilled in creating and formatting documents, presentations, and spreadsheets using Microsoft Word, PowerPoint, and Excel",
        "Self-motivated and well people oriented",
    ]


def test_work_and_training_heading_scopes_experience_and_ignores_award_as_employer():
    text = """CRISTINA O. PAGUIO, LPT
EDUCATION
BALIUAG UNIVERSITY | Bachelor of Elementary Education
Microsoft Excel Specialist (Excel 2019 Associate)
Professional Certification
Licensed Professional Teacher | PRC (March 2025)
WORK AND TRAINING EXPERIENCE
SAINT JOSEPH SCHOOL OF CANDABA
Private Teacher | Full Time 2025-2026
Most Promising Award
Trainings and Seminars
Classroom Management Seminar
"""

    assert extract_education(text) == [{
        "degree": "Bachelor of Elementary Education",
        "institution": "BALIUAG UNIVERSITY",
    }]
    assert extract_experience_records(text) == [{
        "job_title": "Private Teacher",
        "company": "SAINT JOSEPH SCHOOL OF CANDABA",
        "location": "Not Identified",
        "years": 1.0,
    }]
    assert [
        record["certification_name"] for record in extract_certifications(text)
    ] == ["Licensed Professional Teacher"]


def test_ojt_tutor_phone_and_venue_date_metadata_are_reconstructed():
    text = """HAIFA ERICA S. SAMPANG
Contact no.: 0955-129- 0401
EXPERIENCE
On-The-Job Traning KUYA J RESTAURANT (270 hours)
Kuya J branch from November 27, 2023 to January 21, 2024
Home-Based Tutor
(Self- Employed| [River One San Isidro, San Luis, Pampanga] | July 21, 2024- March 2026)
SEMINAR/TRAINING ATTENDED
Strengthening research capabilities through research trends
Venue: Via zoom meeting administered by Bestlink College of the Philippines. Inclusive Date: September 16, 2022
I hereby testify that the above information is true.
SAMPANG, HAIFA ERICA S.
Applicant's Signature
"""

    assert extract_contact_info(text)["phone"] == "0955-129- 0401"
    assert [
        (record["job_title"], record["company"])
        for record in extract_experience_records(text)
    ] == [
        ("On-the-Job Training", "KUYA J RESTAURANT"),
        ("Home-Based Tutor", "Self-Employed"),
    ]
    trainings = extract_certifications(text)
    assert trainings == [{
        "certification_name": "Strengthening research capabilities through research trends",
        "credential_type": "Training",
        "issuer": "Via zoom meeting administered by Bestlink College of the Philippines",
        "date_obtained": "September 16, 2022",
    }]


def test_reference_name_label_cannot_override_header_applicant_name():
    text = """Canasa Camille S.
Email Address: canasacams8@gmail.com
CHARACTER REFERENCE
Name: Vivian D. Dela Victoria
Position: Master Teacher 1
Name: Queenie Mahree D. Sabangan
Position: Teacher 1
Camille S. Canasa
Applicant's Signature
"""

    assert extract_contact_info(text)["name"] == "Camille S. Canasa"
    assert extract_experience_records(text) == []


def test_consecutive_promotions_reuse_only_the_explicit_nearby_employer():
    text = """EXPERIENCE
VILLARICA PAWNSHOP
Branch Associate
December 9, 2025 to March 1, 2026
Cashier (Promoted)
March 2, 2026 to April 2, 2026
REFERENCES
"""

    assert [
        (record["job_title"], record["company"])
        for record in extract_experience_records(text)
    ] == [
        ("Branch Associate", "VILLARICA PAWNSHOP"),
        ("Cashier (Promoted)", "VILLARICA PAWNSHOP"),
    ]
