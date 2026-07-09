import re

from .nlp_pipeline import extract_years_of_experience
from .recommender import SKILL_ALIASES, get_degree_rank


SECTION_RULES = [
    ("Skills", re.compile(r"^\s*(?:technical\s+)?skills?\s*:?", re.I)),
    ("Work Experience", re.compile(
        r"^\s*(?:professional\s+experience|work\s+experience|employment(?:\s+history)?|"
        r"work\s+history|teaching\s+experience|career\s+history)\s*:?", re.I
    )),
    ("Education", re.compile(
        r"^\s*(?:education(?:al)?(?:\s+background|\s+history)?|academic\s+background|"
        r"qualifications?)\s*:?", re.I
    )),
    ("Summary", re.compile(r"^\s*(?:professional\s+)?(?:summary|profile|objective)\s*:?", re.I)),
    ("Certifications", re.compile(r"^\s*(?:professional\s+)?certifications?\s*:?", re.I)),
    ("Projects", re.compile(r"^\s*projects?\s*:?", re.I)),
]

SECTION_PRIORITY = {
    "Skills": 0,
    "Work Experience": 1,
    "Education": 2,
    "Certifications": 3,
    "Projects": 4,
    "Summary": 5,
    "Resume": 6,
}


def _clip_excerpt(value, limit=220):
    value = re.sub(r"\s+", " ", value or "").strip(" -|\t")
    if len(value) <= limit:
        return value
    return value[:limit - 3].rstrip() + "..."


def _resume_lines(text):
    records = []
    current_section = "Resume"
    for index, raw_line in enumerate((text or "").splitlines()):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        for section, pattern in SECTION_RULES:
            if pattern.match(line):
                current_section = section
                break
        records.append({"index": index, "text": line, "section": current_section})
    return records


def _term_found(text, term):
    lowered = text.lower()
    term = term.strip().lower()
    if not term:
        return False
    if re.search(r"[^a-z0-9\s]", term):
        return term in lowered
    pattern = r"\b" + r"\s+".join(re.escape(word) for word in term.split()) + r"\b"
    return bool(re.search(pattern, lowered))


def _skill_evidence(lines, skill):
    variants = {skill.strip().lower()}
    variants.update(SKILL_ALIASES.get(skill.strip().lower(), []))
    candidates = []
    for line in lines:
        if any(_term_found(line["text"], variant) for variant in variants):
            score = SECTION_PRIORITY.get(line["section"], 6)
            candidates.append((score, len(line["text"]), line))
    if not candidates:
        return None
    line = min(candidates, key=lambda item: (item[0], item[1]))[2]
    return {
        "section": line["section"],
        "excerpt": _clip_excerpt(line["text"]),
    }


def _record_evidence(lines, primary_text, secondary_text=None, preferred_section=None):
    terms = [term for term in (primary_text, secondary_text) if term and term not in {
        "Not Identified", "Unknown Institution"
    }]
    candidates = []
    for line in lines:
        if any(term.lower() in line["text"].lower() for term in terms):
            preferred = 0 if line["section"] == preferred_section else 1
            candidates.append((preferred, len(line["text"]), line))
    if not candidates:
        return None

    anchor = min(candidates, key=lambda item: (item[0], item[1]))[2]
    nearby = [anchor["text"]]
    for line in lines:
        if line["index"] <= anchor["index"] or line["index"] > anchor["index"] + 2:
            continue
        if re.match(r"(?i)^(?:responsibilities|duties|environment|project\s+description)\s*:", line["text"]):
            break
        if len(nearby) < 3:
            nearby.append(line["text"])
    return {
        "section": anchor["section"],
        "excerpt": _clip_excerpt(" | ".join(dict.fromkeys(nearby))),
    }


def build_candidate_evidence(resume_text, matched_skills, missing_skills,
                             matched_critical_skills, missing_critical_skills,
                             matched_preferred_skills, experience_records,
                             education_records, experience_requirement=0,
                             education_requirement=None):
    lines = _resume_lines(resume_text)
    matched_critical_skills = matched_critical_skills or []
    matched_skills = matched_skills or []
    matched_preferred_skills = matched_preferred_skills or []

    skill_items = []
    seen_skills = set()
    skill_groups = [
        ("Critical", matched_critical_skills),
        ("Required", matched_skills),
        ("Preferred", matched_preferred_skills),
    ]
    for kind, skills in skill_groups:
        for skill in skills:
            key = skill.strip().lower()
            if not key or key in seen_skills:
                continue
            seen_skills.add(key)
            evidence = _skill_evidence(lines, skill)
            skill_items.append({
                "name": skill,
                "kind": kind,
                "section": evidence["section"] if evidence else "Resume",
                "excerpt": evidence["excerpt"] if evidence else "Matched in the extracted resume text.",
            })

    experience_items = []
    for record in experience_records[:3]:
        evidence = _record_evidence(
            lines, record.company, record.job_title, "Work Experience"
        )
        experience_items.append({
            "job_title": record.job_title,
            "company": record.company,
            "location": record.location,
            "years": record.years,
            "section": evidence["section"] if evidence else "Work Experience",
            "excerpt": evidence["excerpt"] if evidence else "Extracted from the work-history section.",
        })

    education_items = []
    for record in education_records[:3]:
        evidence = _record_evidence(
            lines, record.institution, record.degree, "Education"
        )
        education_items.append({
            "degree": record.degree,
            "institution": record.institution,
            "section": evidence["section"] if evidence else "Education",
            "excerpt": evidence["excerpt"] if evidence else "Extracted from the education section.",
        })

    total_experience = extract_years_of_experience(resume_text)
    experience_status = None
    if experience_requirement and experience_requirement > 0:
        experience_status = {
            "met": total_experience >= experience_requirement,
            "detected": total_experience,
            "required": experience_requirement,
        }

    education_status = None
    required_rank = get_degree_rank(education_requirement)
    if required_rank > 0:
        detected_rank = max(
            (get_degree_rank(record.degree) for record in education_records),
            default=0,
        )
        education_status = {
            "met": detected_rank >= required_rank,
            "required": education_requirement,
        }

    missing_items = []
    for kind, skills in (
        ("Critical", missing_critical_skills or []),
        ("Required", missing_skills or []),
    ):
        for skill in skills:
            missing_items.append({"name": skill, "kind": kind})

    return {
        "skills": skill_items[:8],
        "skill_total": len(skill_items),
        "experience": experience_items,
        "experience_total": len(experience_records),
        "education": education_items,
        "education_total": len(education_records),
        "missing": missing_items,
        "experience_status": experience_status,
        "education_status": education_status,
    }
