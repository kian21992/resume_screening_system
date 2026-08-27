"""Education-domain knowledge shared by resume extraction services.

The module intentionally stays deterministic: it supplements spaCy with an
education ontology, section context, and structural validation. Extractors can
therefore explain why an item was accepted and regression-test the behavior.
"""

import re


EDUCATION_SKILL_CATALOG = {
    "Lesson Planning": ["lesson plans", "lesson preparation", "daily lesson log", "dll"],
    "Classroom Management": ["behavior management", "classroom discipline"],
    "Instructional Strategies": ["teaching strategies", "pedagogy"],
    "Learning Objectives": ["learning outcomes"],
    "Learning Material Development": [
        "instructional materials development", "module development",
    ],
    "Curriculum Development": ["curriculum design", "curriculum planning", "course development"],
    "Curriculum Implementation": ["curriculum delivery"],
    "Differentiated Instruction": ["differentiated learning", "differentiation"],
    "Instructional Design": ["learning design"],
    "Student Assessment": ["learning assessment", "formative assessment", "summative assessment"],
    "Test Construction": ["test item writing", "test development"],
    "Rubric Development": ["rubric design"],
    "Performance-Based Assessment": ["performance assessment"],
    "Diagnostic Assessment": ["diagnostic testing"],
    "Constructive Feedback": [],
    "Progress Monitoring": ["student progress monitoring", "learning progress tracking"],
    "Inclusive Education": ["inclusive teaching", "inclusive classroom"],
    "Special Education": ["sped", "special needs education"],
    "Individualized Education Program": ["individualized education plan", "iep"],
    "Educational Technology": ["edtech", "technology integration", "technology in teaching"],
    "Online Teaching": ["online instruction", "remote teaching", "virtual classroom"],
    "Blended Learning": ["hybrid learning"],
    "Modular Distance Learning": ["modular instruction"],
    "Learning Management Systems": ["lms", "google classroom", "moodle", "canvas lms"],
    "Inquiry-Based Learning": [
        "inquiry-based instruction", "inquiry based teaching",
    ],
    "Project-Based Learning": ["project-based instruction"],
    "Collaborative Learning": ["cooperative learning"],
    "Student-Centered Learning": ["learner-centered teaching"],
    "Outcomes-Based Education": ["obe"],
    "Universal Design for Learning": ["udl"],
    "Remedial Instruction": ["remedial teaching"],
    "Student Engagement": ["learner engagement", "engaging students"],
    "Parent Communication": ["parent-teacher communication", "family communication"],
    "Child Development": ["learner development", "developmental learning"],
    "Literacy Instruction": ["literacy teaching"],
    "Numeracy Instruction": ["numeracy teaching"],
    "Reading Intervention": ["remedial reading"],
    "Multigrade Teaching": ["multi-grade instruction"],
    "Mother Tongue-Based Multilingual Education": ["mtb-mle", "mother tongue instruction"],
    "English Language Teaching": ["english teaching"],
    "Mathematics Instruction": ["math teaching", "teaching mathematics"],
    "Science Instruction": ["science teaching", "teaching science"],
    "Filipino Language Instruction": [
        "filipino teaching", "teaching filipino", "pagtuturo ng filipino",
    ],
    "Araling Panlipunan Instruction": ["araling panlipunan teaching", "ap teaching"],
    "TLE Instruction": ["tle teaching"],
    "MAPEH Instruction": ["mapeh teaching"],
    "Values Education": ["values education teaching", "esp teaching"],
    "Gradebook Management": ["grade computation", "grade recording"],
    "Classroom Record Keeping": ["school forms management"],
    "Student Portfolio Assessment": ["portfolio assessment", "learner portfolio evaluation"],
    "Guidance and Counseling": ["guidance counselling", "student counseling"],
    "Research": ["educational research", "action research"],
    "Mentoring": ["teacher mentoring", "student mentoring"],
    "Facilitation": ["learning facilitation", "facilitating activities"],
}


SECTION_ALIASES = {
    "skills": (
        r"skills?", r"technical skills?", r"core competencies", r"competencies",
        r"areas? of expertise", r"teaching competencies", r"professional skills?",
        r"qualifications? (?:and|&) skills?", r"skills? (?:and|&) abilities",
        r"comp[eé]tences(?: techniques| informatiques| transversales)?",
    ),
    "experience": (
        r"experience", r"experiences", r"work experience", r"work experiences",
        r"working experience", r"working experiences",
        r"professional experience", r"employment history", r"work history",
        r"teaching experience", r"teaching experiences", r"faculty experience",
        r"academic experience", r"education experience", r"career history",
        r"professional background", r"work[- ]related experience", r"positions? held",
        r"work (?:and|&) training experience",
        r"exp[eé]riences? professionnelles?", r"exp[eé]rience professionnelle",
    ),
    "education": (
        r"education", r"educational background", r"education history",
        r"educational history", r"educational qualifications?", r"education details",
        r"educational attainment", r"education attainment", r"professional qualifications?",
        r"academic background", r"academic history", r"academic qualifications?", r"qualifications?",
        r"degrees?", r"scholastic records?", r"formations?", r"parcours acad[eé]mique",
    ),
    "certifications": (
        r"certifications?", r"certificates?", r"licenses?", r"licensure", r"eligibility",
        r"certifications? and licenses?", r"professional certifications?",
    ),
    "training": (
        r"training", r"trainings", r"seminars?", r"professional development",
        r"trainings? attended", r"seminars? attended",
        r"trainings? and seminars?(?: attended)?",
        r"seminars? and trainings?(?: attended)?",
    ),
    "projects": (
        r"projects?", r"academic projects?", r"research projects?", r"coursework",
        r"relevant coursework",
        r"(?:relevant )?coursework and projects?",
    ),
    "awards": (
        r"awards?", r"achievements?", r"honors?", r"distinctions?",
        r"scholarships? and honors?(?: received)?",
    ),
    "languages": (r"languages?", r"langues"),
    "references": (r"references?", r"character references?"),
    "profile": (
        r"summary", r"professional summary", r"profile", r"about me",
        r"objective", r"career objective",
    ),
    "personal": (
        r"personal information", r"personal data", r"personal background",
        r"contact", r"contact information",
    ),
    "affiliations": (r"affiliations?", r"organizations?", r"activities"),
    "interests": (r"interests?", r"hobbies", r"interests? (?:and|&) hobbies"),
}

_SECTION_PATTERNS = {
    name: re.compile(rf"^(?:{'|'.join(aliases)})$", re.IGNORECASE)
    for name, aliases in SECTION_ALIASES.items()
}

_BULLET_RE = re.compile(r"^[\s\-*•▪\u2022\u25aa]+")
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:present|current|till date|to date)\b",
    re.IGNORECASE,
)
_DUTY_START_RE = re.compile(
    r"(?i)^(?:responsible|managed|developed|created|implemented|prepared|"
    r"provided|assisted|collaborated|communicated|conducted|facilitated|"
    r"maintained|organized|coordinated|handled|taught|designed|supported|"
    r"encourag(?:e|ed|ing)|ensur(?:e|ed|ing)|observ(?:e|ed|ing)|"
    r"supervis(?:e|ed|ing)|participat(?:e|ed|ing)|promot(?:e|ed|ing))\b"
)
_EDUCATION_ROLE_RE = re.compile(
    r"(?i)\b(?:teacher|instructor|professor|lecturer|educator|tutor|faculty|"
    r"guidance counselor|school administrator|teaching assistant|practice teacher|"
    r"student teacher|teacher aide|dean|principal)\b"
)
_GRADE_RE = re.compile(
    r"(?i)\b(?:kindergarten|pre[- ]?k|elementary|primary|secondary|senior high|"
    r"junior high|grade(?:s)?\s*(?:[1-9]|1[0-2])(?:\s*(?:-|to)\s*(?:[1-9]|1[0-2]))?)\b"
)
_SUBJECT_RE = re.compile(
    r"(?i)\b(?:english|filipino|mathematics|math|science|biology|chemistry|physics|"
    r"social studies|history|music|physical education|tle|araling panlipunan|"
    r"edukasyon sa pagpapakatao|computer science)\b"
)
_LICENSE_RE = re.compile(
    r"(?i)\b(?:licensed professional teacher|let passer|lpt|teaching license|"
    r"professional teaching certificate)\b"
)


def normalize_heading(line):
    """Return heading-like text without bullets, punctuation, or inline data."""
    value = _BULLET_RE.sub("", line or "").strip()
    value = re.sub(r"^[A-Z]\s*[.)-]\s*", "", value, flags=re.I)
    value = value.split(":", 1)[0]
    value = re.sub(r"\s+", " ", value).strip(" ,/&-|.").casefold()
    return value


def classify_section_heading(line):
    """Classify a standalone or combined resume section heading.

    Combined headings return the first recognized section in document order;
    callers interested in all parts can use ``classify_combined_heading``.
    """
    parts = classify_combined_heading(line)
    return parts[0] if parts else None


def classify_combined_heading(line):
    raw = _BULLET_RE.sub("", line or "").strip()
    raw = re.sub(r"^[A-Z]\s*[.)-]\s*", "", raw, flags=re.I)
    raw = raw.split(":", 1)[0]
    # A few established combined headings have a domain meaning that cannot
    # be inferred by classifying each word around ``and``/``&`` separately.
    # Keep this list narrow so legacy phrases such as ``Scholarships and
    # Honors Received`` retain their established section behavior.
    normalized_raw = normalize_heading(raw)
    if re.fullmatch(r"qualifications? (?:and|&) skills?|skills? (?:and|&) abilities", normalized_raw, re.I):
        return ["skills"]
    if re.fullmatch(r"work (?:and|&) training experience", normalized_raw, re.I):
        return ["experience"]
    if re.fullmatch(r"interests? (?:and|&) hobbies", normalized_raw, re.I):
        return ["interests"]
    parts = [part.strip() for part in re.split(r"\s*(?:,|&|/|\band\b)\s*", raw, flags=re.I) if part.strip()]
    classified = []
    unmatched = False
    for part in parts:
        normalized = normalize_heading(part)
        match = next((name for name, pattern in _SECTION_PATTERNS.items() if pattern.fullmatch(normalized)), None)
        if match and match not in classified:
            classified.append(match)
        elif not match:
            unmatched = True
    # A combined heading is trustworthy only when every segment is a known
    # heading. This prevents skill text such as "Activities/Game" from being
    # interpreted as the ACTIVITIES section.
    if len(parts) > 1 and unmatched:
        return []
    return classified


def segment_resume(text):
    """Return ordered section blocks with source lines preserved."""
    sections = []
    current = {"section": "header", "heading": None, "lines": []}
    for line_number, raw_line in enumerate((text or "").splitlines(), start=1):
        section = classify_section_heading(raw_line)
        if section:
            if current["lines"] or current["heading"]:
                sections.append(current)
            current = {
                "section": section,
                "heading": raw_line.strip(),
                "line_number": line_number,
                "lines": [],
            }
        else:
            current["lines"].append(raw_line)
    if current["lines"] or current["heading"]:
        sections.append(current)
    return sections


def education_skill_vocabulary():
    return {
        canonical: list(dict.fromkeys([canonical, *aliases]))
        for canonical, aliases in EDUCATION_SKILL_CATALOG.items()
    }


def extract_education_entities(text):
    """Hybrid rule/ontology recognizer with evidence and confidence metadata."""
    entities = []
    seen = set()
    current_section = "header"
    vocabulary = education_skill_vocabulary()

    def add(kind, name, evidence, line_number, confidence):
        key = (kind, name.casefold(), evidence.casefold())
        if key in seen:
            return
        seen.add(key)
        entities.append({
            "type": kind,
            "name": name,
            "evidence": evidence,
            "section": current_section,
            "line_number": line_number,
            "confidence": confidence,
        })

    for line_number, raw_line in enumerate((text or "").splitlines(), start=1):
        line = raw_line.strip()
        section = classify_section_heading(line)
        if section:
            current_section = section
            continue
        if not line:
            continue
        for match in _EDUCATION_ROLE_RE.finditer(line):
            add("education_role", match.group(0), line, line_number, "high" if current_section == "experience" else "medium")
        for match in _GRADE_RE.finditer(line):
            add("grade_level", match.group(0), line, line_number, "high" if current_section in {"experience", "education"} else "medium")
        for match in _SUBJECT_RE.finditer(line):
            add("subject", match.group(0), line, line_number, "medium")
        for match in _LICENSE_RE.finditer(line):
            add("license", match.group(0), line, line_number, "high")
        for canonical, variants in vocabulary.items():
            if any(re.search(r"\b" + re.escape(variant) + r"\b", line, re.I) for variant in variants):
                confidence = "high" if current_section == "skills" else "medium"
                add("education_skill", canonical, line, line_number, confidence)
    return entities


def validate_experience_record(record):
    """Return ``(is_valid, reason)`` for a structured experience record."""
    title = re.sub(r"\s+", " ", (record.get("job_title") or "")).strip()
    company = re.sub(r"\s+", " ", (record.get("company") or "")).strip()
    years = record.get("years")
    if not title:
        return False, "missing role"
    if len(title) > 100 or classify_section_heading(title):
        return False, "role resembles a heading"
    if _DUTY_START_RE.match(title) and len(title.split()) > 4:
        return False, "role resembles a responsibility"
    if not company:
        return False, "missing employer"
    if company != "Not Identified" and (re.match(r"^[\W_]", company) or "@" in company):
        return False, "malformed employer"
    try:
        duration = float(years or 0)
        duration_unknown = bool(record.get("duration_unknown"))
        if duration < 0 or duration > 50 or (duration == 0 and not duration_unknown):
            return False, "invalid duration"
    except (TypeError, ValueError):
        return False, "invalid duration"
    return True, None


def validate_education_record(record):
    """Return ``(is_valid, reason)`` for a structured education record."""
    degree = re.sub(r"\s+", " ", (record.get("degree") or "")).strip()
    institution = re.sub(r"\s+", " ", (record.get("institution") or "")).strip()
    if not degree or len(degree) > 255:
        return False, "missing or oversized credential"
    if classify_section_heading(degree) or _DUTY_START_RE.match(degree):
        return False, "credential resembles non-education content"
    if re.search(r"(?i)\b(?:awards?|honors?|responsibilities|duties|skills?)\b", degree):
        return False, "credential contains unrelated content"
    if not institution:
        return False, "missing institution value"
    if institution != "Unknown Institution" and ("@" in institution or _DATE_RE.fullmatch(institution)):
        return False, "invalid institution"
    return True, None
