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

DATE_EVIDENCE_RE = re.compile(
    r"\b(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[\s,./-]+)?(?:19|20)?\d{2}\s*(?:-|–|—|to)\s*"
    r"(?:present|current|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)[\s,./-]+)?(?:19|20)?\d{2}\b|\b(?:19|20)\d{2}\b",
    re.I,
)

EVIDENCE_NOISE_RE = re.compile(
    r"^(?:responsibilities|duties|environment|technologies|tools|skills?|"
    r"project\s+description|references?)\s*:|"
    r"^(?:responsible\s+for|developed|managed|created|implemented|assisted|handled|"
    r"prepared|collaborated|worked|provided|performed|designed|led)\b",
    re.I,
)


def _clip_excerpt(value, limit=220):
    value = re.sub(r"\s+", " ", value or "").strip(" -|\t")
    if len(value) <= limit:
        return value
    return value[:limit - 3].rstrip() + "..."


def _resume_lines(text):
    records = []
    current_section = "Resume"
    block = 0
    for index, raw_line in enumerate((text or "").splitlines()):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            block += 1
            continue
        heading_section = None
        for section, pattern in SECTION_RULES:
            if pattern.match(line):
                current_section = section
                heading_section = section
                break
        records.append({
            "index": index,
            "text": line,
            "section": current_section,
            "block": block,
            "is_heading": heading_section is not None and bool(re.fullmatch(
                r"[A-Za-z &/]+:?", line
            )),
        })
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


def _record_evidence(lines, primary_text, secondary_text=None, preferred_section=None,
                     evidence_kind=None):
    invalid_values = {"Not Identified", "Unknown Institution"}
    primary = primary_text if primary_text and primary_text not in invalid_values else None
    secondary = secondary_text if secondary_text and secondary_text not in invalid_values else None

    def field_match(line, value):
        return bool(value and _term_found(line["text"], value))

    # Company/institution is the strongest record identity. Generic secondary
    # values such as "Teacher" or "High School" can occur in nearby records,
    # so use them as the anchor only when the primary value is unavailable.
    primary_candidates = [line for line in lines if field_match(line, primary)]
    anchor_pool = primary_candidates or [
        line for line in lines if field_match(line, secondary)
    ]
    candidates = []
    for line in anchor_pool:
        preferred = 0 if line["section"] == preferred_section else 1
        candidates.append((preferred, len(line["text"]), line))
    if not candidates:
        return None

    anchor = min(candidates, key=lambda item: item[:2])[2]
    same_context = [line for line in lines if (
        line["section"] == anchor["section"]
        and line["block"] == anchor["block"]
        and not line["is_heading"]
    )]
    nearby = [anchor["text"]]

    # Add the secondary field only when it is an exact field line (possibly
    # accompanied by a date or label). This prevents a generic title from
    # matching a different nearby record, e.g. "Teacher" inside "Senior High
    # School Teacher".
    secondary_matches = []
    for line in same_context:
        if not field_match(line, secondary) or EVIDENCE_NOISE_RE.search(line["text"]):
            continue
        without_date = DATE_EVIDENCE_RE.sub("", line["text"])
        without_label = re.sub(
            r"(?i)^(?:role|position|job\s*title|designation|degree|course|program)\s*:\s*",
            "",
            without_date,
        ).strip(" ()[]-|,:.")
        exact_secondary = without_label.lower() == secondary.lower().strip()
        contains_primary = field_match(line, primary)
        if exact_secondary or contains_primary:
            secondary_matches.append(line)
    if secondary_matches:
        secondary_line = min(
            secondary_matches,
            key=lambda line: (abs(line["index"] - anchor["index"]), len(line["text"])),
        )
        if secondary_line["text"] not in nearby:
            nearby.append(secondary_line["text"])
    else:
        secondary_line = None

    # Dates are useful evidence, but date rows often contain awards or honors.
    # Keep only the date token unless the row is already one of the exact field
    # lines selected above.
    dated_lines = [
        line for line in same_context
        if DATE_EVIDENCE_RE.search(line["text"])
        and abs(line["index"] - anchor["index"]) <= 2
    ]
    if dated_lines:
        selected_indices = [anchor["index"]]
        if secondary_line is not None:
            selected_indices.append(secondary_line["index"])
        first_field = min(selected_indices)
        last_field = max(selected_indices)

        def date_rank(line):
            distance = min(abs(line["index"] - index) for index in selected_indices)
            # Conventional layouts normally put education dates before the
            # degree/school and employment dates after the employer/title.
            # This directional tie-break prevents a neighboring record's date
            # from winning when both are equally close.
            if evidence_kind == "education":
                direction_penalty = 0 if line["index"] <= first_field else 1
            elif evidence_kind == "work":
                direction_penalty = 0 if line["index"] >= last_field else 1
            else:
                direction_penalty = 0
            return distance, direction_penalty

        date_line = min(dated_lines, key=date_rank)
        date_match = DATE_EVIDENCE_RE.search(date_line["text"])
        date_evidence = (
            date_line["text"] if date_line["text"] in nearby else date_match.group(0)
        )
        if date_evidence not in nearby:
            nearby.append(date_evidence)

    if not nearby:
        nearby = [anchor["text"]]
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
            lines, record.company, record.job_title, "Work Experience", "work"
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
            lines, record.institution, record.degree, "Education", "education"
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
