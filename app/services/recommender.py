def generate_recommendation(fit_score, min_fit_score=50.0):
    """
    Generates a recommendation label based on the fit score.
    Logic:
    - Qualified: >= 75%
    - For Review: >= min_fit_score (e.g., 50%) but < 75%
    - Not Qualified: < min_fit_score
    """
    if fit_score >= 75.0:
        return "Qualified"
    elif fit_score >= min_fit_score:
        return "For Review"
    else:
        return "Not Qualified"

import re as _re
import hashlib
from rapidfuzz import fuzz
from nltk.stem import PorterStemmer
from app.services.education_domain import (
    classify_combined_heading,
    classify_section_heading,
    education_skill_vocabulary,
)

_STEMMER = PorterStemmer()


def _stem_tokens(text):
    """Lowercase, tokenize on alphanumerics, and Porter-stem each token."""
    return [_STEMMER.stem(t) for t in _re.findall(r'[a-z0-9]+', text.lower())]

SKILL_ALIASES = {
    "javascript": ["js", "ecmascript", "es6"],
    "js": ["javascript", "ecmascript", "es6"],
    "react": ["react.js", "reactjs"],
    "node": ["node.js", "nodejs"],
    "node.js": ["node", "nodejs"],
    "html": ["html5"],
    "css": ["css3"],
    "python": ["python3", "python 3"],
    "postgres": ["postgresql"],
    "postgresql": ["postgres", "postgre sql"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "natural language processing": ["nlp"],
    "lesson planning": ["lesson plan", "lesson plans", "develop lesson plans", "developlessonplans", "curriculum planning", "prepared and delivered lessons", "designs engaging lesson plans"],
    "classroom management": ["classroommanagement", "classroom management strategies", "managed classroom instruction"],
    "communication skills": ["communication skill", "written and verbal communication", "verbal and written communication skills"],
    "c#": ["c sharp"],
    "c sharp": ["c#"],
}

# Canonical, resume-independent vocabulary used by ``extract_resume_skills``.
# Job skills are added to this vocabulary at evaluation time, but the extractor
# does not depend on them: it can still identify skills that were never listed
# in the job posting.
RESUME_SKILL_CATALOG = {
    "Python": ["python3", "python 3"],
    "Java": [],
    "JavaScript": ["js", "ecmascript", "es6"],
    "TypeScript": [],
    "C": [],
    "C++": ["cpp"],
    "C#": ["c sharp"],
    "Go": ["golang"],
    "R": [],
    "PHP": [],
    "Ruby": [],
    "Kotlin": [],
    "Swift": [],
    "HTML": ["html5"],
    "CSS": ["css3"],
    "SQL": [],
    "PostgreSQL": ["postgres", "postgre sql"],
    "MySQL": [],
    "SQLite": [],
    "MongoDB": [],
    "Oracle": [],
    "Django": [],
    "Flask": [],
    "FastAPI": [],
    "React": ["react.js", "reactjs"],
    "Angular": [],
    "Vue.js": ["vue", "vuejs"],
    "Node.js": ["node", "nodejs"],
    ".NET": ["dotnet", "asp.net"],
    "Spring Boot": [],
    "REST API": ["restful api", "rest apis", "restful services"],
    "Git": [],
    "GitHub": [],
    "Docker": [],
    "Kubernetes": ["k8s"],
    "AWS": ["amazon web services"],
    "Microsoft Azure": ["azure"],
    "Google Cloud Platform": ["gcp", "google cloud"],
    "Linux": [],
    "DevOps": [],
    "CI/CD": ["continuous integration", "continuous deployment"],
    "Machine Learning": ["ml"],
    "Artificial Intelligence": ["ai"],
    "Natural Language Processing": ["nlp"],
    "Deep Learning": [],
    "Data Analysis": ["data analytics"],
    "Data Visualization": [],
    "Pandas": [],
    "NumPy": ["numpy"],
    "Scikit-learn": ["sklearn", "scikit learn"],
    "TensorFlow": [],
    "PyTorch": [],
    "Power BI": ["powerbi"],
    "Tableau": [],
    "Microsoft Excel": ["ms excel", "excel"],
    "Microsoft Office": ["ms office"],
    "Project Management": [],
    "Agile": [],
    "Scrum": [],
    "Communication": ["communication skills", "written and verbal communication"],
    "Leadership": ["leadership skills"],
    "Teamwork": ["team collaboration", "collaboration skills"],
    "Problem Solving": ["problem-solving", "problem solving skills"],
    "Critical Thinking": [],
    "Time Management": [],
    "Lesson Planning": SKILL_ALIASES["lesson planning"],
    "Classroom Management": SKILL_ALIASES["classroom management"],
    "Curriculum Development": ["curriculum design"],
    "Teaching": ["instruction", "instructional delivery"],
    "Assessment": ["student assessment", "learning assessment"],
    "Differentiated Instruction": [],
    "Research": ["research skills"],
    "Counseling": ["counselling"],
    "Case Management": [],
    "Customer Service": ["customer support"],
    "Technical Support": ["it support"],
    "Accounting": [],
    "Bookkeeping": [],
    "Payroll": [],
    "Recruitment": ["recruiting", "talent acquisition"],
    "Human Resources": ["hr management"],
}

_SKILL_SECTION_HEADER_RE = _re.compile(
    r"^\s*(?:technical\s+skills?|professional\s+skills?|personal\s+skills?"
    r"|key\s+skills?|relevant\s+skills?|special\s+skills?|skills?"
    r"|skills?\s*(?:&|and|/)\s*(?:competenc(?:y|ies)|abilities|qualities"
    r"|attributes?|strengths?|interests?|expertise)"
    r"|skills?\s+highlights?|knowledge,?\s+skills?\s+and\s+abilities"
    r"|core\s+competenc(?:y|ies)|competenc(?:y|ies)|areas?\s+of\s+expertise"
    r"|technical\s+proficienc(?:y|ies)|technologies|tools)\s*"
    r"(?::\s*(?P<content>.*))?$",
    _re.IGNORECASE,
)
_NON_SKILL_SECTION_RE = _re.compile(
    r"^\s*(?:objective|career\s+objective|summary|profile|professional\s+summary|contact(?:\s+information)?"
    r"|education(?:al\s+(?:background|attainment))?|academic\s+background"
    r"|academic\s+(?:credentials?|qualifications?)|educational\s+qualifications?"
    r"|work\s+experience|professional\s+experience|experience"
    r"|employment(?:\s+history)?|projects?|certifications?|licenses?|trainings?"
    r"|professional\s+trainings?(?:\s+and\s+certifications?)?"
    r"|seminars?|awards?|achievements?|references?|personal\s+(?:information|data)"
    r"|interests?|activities|affiliations?|publications?|volunteer(?:ing)?(?:\s+experience)?"
    r"|organizations?|professional\s+development)"
    r"(?:\s*:\s*.*)?\s*$",
    _re.IGNORECASE,
)
_SKILL_CATEGORY_RE = _re.compile(
    r"^(?:programming\s+languages?|languages?|frameworks?|libraries|databases?"
    r"|frameworks?\s*(?:&|and|/)\s*libraries"
    r"|backend\s*(?:&|and|/)\s*databases?"
    r"|tools?\s*(?:&|and|/)\s*platforms?"
    r"|core\s+concepts?"
    r"|cloud(?:\s+platforms?)?|platforms?|software|applications?|methodologies"
    r"|soft\s+skills?|hard\s+skills?|technical|laboratory|others?|tools"
    r"|additional\s+skills?"
    r"|java(?:/j2ee)?\s+technologies|web\s+(?:technologies|development)"
    r"|xml\s+(?:technologies|processing)|design\s+(?:patterns?|methodologies)"
    r"|methodologies/design\s+patterns?"
    r"|operating\s+systems?|version\s+control|cloud\s+technologies"
    r"|testing(?:\s+and\s+logging)?\s+(?:frameworks?|tools?|technologies)"
    r"|database\s+technologies|messaging\s+tech(?:nologies|ologies)"
    r"|ide\s*s(?:\s*/\s*tools?)?)\s*$",
    _re.IGNORECASE,
)
_SKILL_CATEGORY_PREFIX_RE = _re.compile(
    r"^(?:programming\s+languages?|mark-up/xml\s+technologies|tools?\s*&\s*frameworks?"
    r"|java\s+technologies|web\s+technologies|xml\s+technologies|build\s+(?:tools?|automation)"
    r"|web\s+services?|cloud\s+technologies|application/web\s+servers?"
    r"|web/app\s+servers?|application\s+servers?|web\s+servers?|databases?(?:\s+and\s+tools?)?"
    r"|ide(?:s)?\s*/\s*tools?|ide(?:s|\s*s|\s+tools?)"
    r"|operating\s+systems?|design\s+patterns?"
    r"|testing\s+(?:frameworks?|tools?|technologies)(?:\s*/\s*others?)?"
    r"|version\s+control(?:\s+tools?)?"
    r"|change\s+management\s+tools?|scripting(?:/gui)?\s+tools?"
    r"|modeling/\s*case\s+tools?|reporting\s+tools?|designing\s+tools?"
    r"|big\s+data\s+ecosystem|frameworks?"
    r"|orm(?:\s+technologies)?|messaging\s+(?:systems?|tech(?:nologies|ologies))"
    r"|routing\s+technologies|caching\s+technologies|development\s+methodologies"
    r"|methodologies/design\s+patterns?|os\s*&\s*environment|additional\s+skills?"
    r"|web\s+service\s+specifications\s+and\s+implementations"
    r"|project\s+management\s+tools?|manual\s+test\s*&\s*automation\s+tools?"
    r"|release\s*&\s*deployment\s+tools?|frontend|db|platforms?|domain"
    r"|soft\s+skills?|hard\s+skills?|core\s+concepts?"
    r"|frameworks?\s*(?:&|and|/)\s*libraries"
    r"|backend\s*(?:&|and|/)\s*databases?"
    r"|tools?\s*(?:&|and|/)\s*platforms?|tools?)"
    r"\s*(?::|\s{2,}|\t|\s+(?=(?-i:[A-Z0-9])|[.#/+]))\s*",
    _re.IGNORECASE,
)
_CONCATENATED_SKILL_START_RE = _re.compile(
    r"(?<=[a-z0-9)])\s+(?=(?:Ability\s+to|Adapts?|Applies|Collaborates?|Communicates?"
    r"|Creates?|Demonstrates?|Designs?|Develops?|Facilitates?|Implements?|Leads?"
    r"|Leverages?|Maintains?|Manages?|Organizes?|Performs?|Plans?|Possesses?"
    r"|Prepares?|Provides?|Skilled\s+in|Proficient\s+in|Strong\s+knowledge\s+of)"
    r"\b(?=\s+[a-z]))"
)
_COMPETENCY_PHRASE_RE = _re.compile(
    r"^(?:Ability\s+to|Adapts?|Applies|Collaborates?|Communicates?|Creates?"
    r"|Demonstrates?|Designs?|Develops?|Facilitates?|Implements?|Leads?|Leverages?"
    r"|Maintains?|Manages?|Organizes?|Performs?|Plans?|Possesses?|Prepares?"
    r"|Provides?|Conducts?|Skill(?:ed|ful)|Skilful|Proficient\s+in|Strong\b"
    r"|Excellent\b|Effective\b|Knowledge\b|Experienced?\b"
    r"|Expertise\b|Extensive\s+expertise|Familiar\b|Highly\s+capable"
    r"|Well\s+acquainted|Solid\s+understanding|High\s+degree|Good\s+knowledge"
    r"|Acted\s+as)\b",
    _re.IGNORECASE,
)
_NEGATED_SKILL_RE = _re.compile(
    r"\b(?:no|not|without|lack(?:s|ing)?|unfamiliar\s+with)\b[^.;:]{0,35}$",
    _re.IGNORECASE,
)
_ASPIRATIONAL_SKILL_RE = _re.compile(
    r"\b(?:interested\s+in|willing\s+to|eager\s+to|hoping\s+to|seeking\s+to|"
    r"would\s+like\s+to|currently\s+learning|planning\s+to\s+learn|"
    r"want(?:s|ing)?\s+to\s+learn)\b[^.;:]{0,50}$",
    _re.IGNORECASE,
)
_CONTEXT_EXCLUDED_HEADER_RE = _re.compile(
    r"^\s*(?:education(?:al\s+(?:background|attainment|qualifications?))?"
    r"|academic\s+(?:background|credentials?|qualifications?)"
    r"|personal\s+(?:information|data|details)|(?:character\s+)?references?|interests?|hobbies"
    r"|awards?|achievements?|affiliations?|professional\s*&\s*civic\s+affiliation"
    r"|certifications?|licenses?|trainings?|seminars?|professional\s+development"
    r"|publications?(?:\s*&\s*presentations?)?)"
    r"(?:\s*:\s*.*)?\s*$",
    _re.IGNORECASE,
)
_CONTEXT_INCLUDED_HEADER_RE = _re.compile(
    r"^\s*(?:objective|career\s+objective|summary|profile|professional\s+summary"
    r"|work\s+experience|professional\s+experience|teaching\s+experience"
    r"|employment(?:\s+history)?|experience|projects?)"
    r"(?:\s*:\s*.*)?\s*$",
    _re.IGNORECASE,
)


def _resume_skill_vocabulary(additional_skills=None):
    vocabulary = {
        canonical: list(dict.fromkeys([canonical, *aliases]))
        for canonical, aliases in RESUME_SKILL_CATALOG.items()
    }
    for canonical, variants in education_skill_vocabulary().items():
        vocabulary[canonical] = list(dict.fromkeys([
            *vocabulary.get(canonical, []),
            *variants,
        ]))
    for raw_skill in additional_skills or []:
        skill = (raw_skill or "").strip()
        if not skill:
            continue
        skill_lower = skill.lower()
        existing = next(
            (
                name for name, variants in vocabulary.items()
                if name.lower() == skill_lower
                or any(variant.lower() == skill_lower for variant in variants)
            ),
            None,
        )
        if existing:
            vocabulary[existing] = list(dict.fromkeys([
                *vocabulary[existing],
                skill,
                *SKILL_ALIASES.get(skill_lower, []),
            ]))
            continue
        aliases = SKILL_ALIASES.get(skill_lower, [])
        vocabulary[skill] = list(dict.fromkeys([skill, *aliases]))
    return vocabulary


def _variant_matches(line, variant):
    flags = 0 if len(variant.strip()) == 1 else _re.IGNORECASE
    if _re.search(r"[^a-z0-9\s]", variant, _re.I):
        pattern = _re.escape(variant)
    else:
        words = variant.split()
        pattern = r"\b" + r"\s+".join(_re.escape(word) for word in words) + r"\b"
    return _re.finditer(pattern, line, flags)


def _positive_skill_mention(text, variants):
    """Return the resume's exact skill wording outside nearby negation."""
    for line in (text or "").splitlines():
        for variant in sorted(variants, key=len, reverse=True):
            for match in _variant_matches(line, variant):
                prefix = line[:match.start()]
                if (
                    not _NEGATED_SKILL_RE.search(prefix)
                    and not _ASPIRATIONAL_SKILL_RE.search(prefix)
                ):
                    return match.group(0)
    return None


def _contextual_skill_text(text):
    """Return resume areas that can support an inferred skill claim.

    Explicit skill sections are handled separately. Education, personal data,
    references, hobbies, awards, and publications are excluded so a technology
    in a degree title or an interest is not automatically treated as applied
    competence. Unheaded text remains available for non-template resumes.
    """
    included = True
    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _re.search(r'(?i)https?://|www\.|\S+@\S+|\b\w+\.(?:com|net|org|edu)(?:/|\b)', line):
            continue
        if _SKILL_SECTION_HEADER_RE.match(line):
            included = False
            continue
        if _CONTEXT_EXCLUDED_HEADER_RE.match(line):
            included = False
            continue
        if _CONTEXT_INCLUDED_HEADER_RE.match(line):
            included = True
            # Inline content after a recognized heading may contain evidence.
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                lines.append(inline)
            continue
        if included:
            lines.append(line)
    return "\n".join(lines)


def _catalog_skills_covered_by_explicit_items(items, vocabulary):
    """Map explicit resume wording to canonical IDs for alias deduplication."""
    covered = set()
    for item in items:
        item_lower = item.lower()
        for canonical, variants in vocabulary.items():
            literal_match = any(
                _literal_skill_found(item_lower, variant.lower())
                for variant in variants
            )
            canonical_stems = {
                stem for stem in _stem_tokens(canonical)
                if len(stem) >= 4
            }
            stem_match = bool(
                canonical_stems
                and canonical_stems.issubset(set(_stem_tokens(item)))
            )
            if literal_match or stem_match:
                covered.add(canonical.lower())
    return covered


def _valid_section_skill(value):
    # Explicit skill sections are trusted more than narrative text. Long
    # competency statements are valid resume skills and technology versions
    # may contain years (for example "Windows 2000/NT/XP"), so neither should
    # be rejected here.
    if not value or len(value) > 300 or len(value.split()) > 50:
        return False
    if _re.search(r"(?:@|https?://|www\.)", value, _re.I):
        return False
    if _re.fullmatch(r"[\W\d_]+", value):
        return False
    if _re.match(r"(?i)^(?:for|and|or|to|including|catering\s+to)\b", value):
        return False
    return True


def _split_top_level_skill_items(value):
    """Split list delimiters while preserving commas inside parentheses."""
    value = value.strip()
    if not value:
        return []
    competency_phrase = bool(_COMPETENCY_PHRASE_RE.match(value))
    structural_items = []
    current = []
    depth = 0
    index = 0
    while index < len(value):
        char = value[index]
        if char in "([{" :
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)

        wide_space = char.isspace() and (
            char == "\t" or (index + 1 < len(value) and value[index + 1].isspace())
        ) and not (index > 0 and value[index - 1] in ",;:")
        spaced_hyphen = (
            char == "-" and index > 0 and index + 1 < len(value)
            and value[index - 1].isspace() and value[index + 1].isspace()
            and (
                "," in value[index + 1:]
                or len(value[:index].split()) >= 4
                or len(_re.findall(r"\s+-\s+", value)) >= 2
            )
        )
        delimiter = depth == 0 and (
            char in "|•·" or (char == ";" and not competency_phrase)
            or wide_space or spaced_hyphen
        )
        if delimiter:
            item = "".join(current).strip(" -*•|,;:")
            if item:
                structural_items.append(item)
            current = []
            if wide_space:
                while index + 1 < len(value) and value[index + 1].isspace():
                    index += 1
        else:
            current.append(char)
        index += 1
    item = "".join(current).strip(" -*•|,;:")
    if item:
        structural_items.append(item)

    items = []
    for structural_item in structural_items:
        first_words = len(structural_item.split(",", 1)[0].split())
        comma_phrase = bool(
            _COMPETENCY_PHRASE_RE.match(structural_item)
            or (
                _re.search(r",\s*(?:and|or)\b", structural_item, _re.I)
                and first_words >= 2
            )
        )
        if comma_phrase or "," not in structural_item:
            items.append(structural_item)
            continue

        current = []
        depth = 0
        for char in structural_item:
            if char in "([{" :
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            if char == "," and depth == 0:
                item = _re.sub(
                    r"^(?:and|or)\s+", "", "".join(current).strip(" ,"),
                    flags=_re.IGNORECASE,
                )
                if item:
                    items.append(item)
                current = []
            else:
                current.append(char)
        item = _re.sub(
            r"^(?:and|or)\s+", "", "".join(current).strip(" ,"),
            flags=_re.IGNORECASE,
        )
        if item:
            items.append(item)
    return items


def _skill_section_items(line):
    """Return display-ready items from one explicit skill-section row."""
    line = _re.sub(r"^\s*[-*•·\uf06c\uf0b7]\s*", "", line or "").strip()
    if not line:
        return []

    # Remove a wide first column only when it is a recognized category. A
    # generic rule would lose real skills in rows such as "Python    Docker".
    wide_columns = [part.strip() for part in _re.split(r"\t+|\s{2,}", line) if part.strip()]
    if (
        len(wide_columns) >= 2
        and (
            _SKILL_CATEGORY_RE.fullmatch(wide_columns[0])
            or _SKILL_CATEGORY_PREFIX_RE.fullmatch(wide_columns[0])
        )
    ):
        line = "  ".join(wide_columns[1:])

    labelled = _re.match(r"^([^:]{1,55}):\s*(.+)$", line)
    if labelled and len(labelled.group(1).split()) <= 6:
        label = labelled.group(1).strip()
        content = labelled.group(2).strip()
        if (
            _SKILL_CATEGORY_RE.fullmatch(label)
            or _SKILL_CATEGORY_PREFIX_RE.fullmatch(label)
        ):
            line = content
        else:
            # Labels such as "Spring Framework" and "NoSQL" are themselves
            # skills; retain them alongside the values that follow.
            line = f"{label}, {content}"
    else:
        line = _SKILL_CATEGORY_PREFIX_RE.sub("", line, count=1).strip()
    if not line or _SKILL_CATEGORY_RE.fullmatch(line):
        return []

    # Some visually separated PDF columns are flattened onto one line. Resume
    # competency sentences commonly reveal the lost boundary through a new
    # capitalized action phrase ("... curriculum Communicates clearly ...").
    chunks = _CONCATENATED_SKILL_START_RE.split(line)
    items = []
    for chunk in chunks:
        items.extend(_split_top_level_skill_items(chunk))
    return [item for item in items if _valid_section_skill(item)]


_SKILL_BULLET_RE = _re.compile(r"^\s*[-*•·\uf06c\uf0b7]\s*")
_WRAPPED_SKILL_PREFIX_RE = _re.compile(
    r"^(?:curriculum\s+and|verbal\s+and|modifying\b|integrating\b|utilizing\b"
    r"|providing\b|excellent\s+in\b)",
    _re.IGNORECASE,
)
_WRAPPED_SKILL_END_RE = _re.compile(
    r"\b(?:and|or|of|in|for|with|to|subject|written|instructional|interactive"
    r"|teaching|formative)\s*$",
    _re.IGNORECASE,
)


def _should_join_wrapped_skill(previous, continuation, joined_lines=0):
    """Detect a PDF-wrapped continuation of an explicit bullet item."""
    previous = _SKILL_BULLET_RE.sub("", previous or "").strip()
    continuation = (continuation or "").strip()
    if not previous or not continuation or len(continuation.split()) > 5:
        return False
    if _SKILL_SECTION_HEADER_RE.match(continuation) or _NON_SKILL_SECTION_RE.match(continuation):
        return False
    if _WRAPPED_SKILL_END_RE.search(previous):
        return True
    if not _WRAPPED_SKILL_PREFIX_RE.match(previous):
        return False
    # Most wrapped bullets use one continuation line. Phrases beginning with
    # "Integrating" commonly wrap across two ("... Teaching and Learning" +
    # "Processes"). Limiting depth prevents the next standalone skill from
    # being swallowed after the phrase is complete.
    max_joined_lines = 2 if _re.match(r"(?i)^integrating\b", previous) else 1
    return joined_lines < max_joined_lines


def extract_resume_skills(resume_text, additional_skills=None):
    """Extract a resume-worded skill inventory independently of job matching.

    Every clean item in an explicit skills/competencies section is preserved as
    written, including longer competency phrases and skills outside the built-in
    vocabulary. Elsewhere, vocabulary-backed mentions are returned using their
    actual resume wording. Alias normalization remains internal to job matching.
    The returned list is informational and does not affect scoring.
    """
    text = resume_text or ""
    text = _re.sub(
        r'(?m)^([^\n]{3,60}(?:&|and)\s+[A-Za-z]+)\n(?=[a-z])',
        r'\1 ',
        text,
    )
    vocabulary = _resume_skill_vocabulary(additional_skills)
    extracted = []
    seen = set()

    def dedupe_key(value):
        tokens = _re.findall(r"[a-z0-9+#.]+", value.lower())
        # These suffixes describe an item rather than distinguish the skill.
        while tokens and tokens[-1] in {"skill", "skills", "proficiency"}:
            tokens.pop()
        return " ".join(tokens)

    def add(value):
        normalized = _re.sub(r"\s+", " ", (value or "")).strip(" -*•|,;:.")
        key = dedupe_key(normalized)
        if normalized and key not in seen:
            seen.add(key)
            extracted.append(normalized)

    def add_section_line(value):
        for part in _skill_section_items(value):
            if _re.match(
                r"(?i)^(?:no|not|without|lack(?:s|ing)?|unfamiliar\s+with)\b",
                part,
            ):
                continue
            add(part)

    # Combined sections often keep skills on an inline labelled bullet rather
    # than under a standalone SKILLS heading. Capture that list and its wrapped
    # continuation lines independently of the surrounding section title.
    text_lines = text.splitlines()
    for index, raw_line in enumerate(text_lines):
        labelled = _re.match(
            r"(?i)^\s*[-*•▪\u2022\u25aa]?\s*skills?\s*:\s*(.+)$",
            raw_line.strip(),
        )
        if not labelled:
            continue
        value = labelled.group(1).strip()
        for lookahead in range(index + 1, min(len(text_lines), index + 3)):
            continuation = text_lines[lookahead].strip()
            if not continuation or _re.match(
                r"(?i)^\s*[-*•▪\u2022\u25aa]?\s*(?:skills?|certifications?|awards?)\s*:",
                continuation,
            ) or _NON_SKILL_SECTION_RE.match(continuation):
                break
            if _SKILL_BULLET_RE.match(continuation):
                break
            value += " " + continuation
        add_section_line(value)

    # Rejoin category rows that wrapped during PDF extraction. Keeping the
    # complete logical row preserves compound skills such as "API Integration"
    # and parenthesized technology lists split across physical lines.
    rebuilt_lines = []
    pending_category_line = None
    rebuilding_skill_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        header = _SKILL_SECTION_HEADER_RE.match(line)
        boundary = classify_section_heading(line)
        if header:
            if pending_category_line:
                rebuilt_lines.append(pending_category_line)
                pending_category_line = None
            rebuilding_skill_section = True
            rebuilt_lines.append(raw_line)
            continue
        if rebuilding_skill_section and boundary and boundary != "skills":
            if pending_category_line:
                rebuilt_lines.append(pending_category_line)
                pending_category_line = None
            rebuilding_skill_section = False
            rebuilt_lines.append(raw_line)
            continue
        if rebuilding_skill_section:
            labelled = _re.match(r"^([^:]{1,55}):\s*(.+)$", line)
            prefixed_category = _SKILL_CATEGORY_PREFIX_RE.match(line)
            is_category_row = bool(
                (
                    labelled
                    and (
                        _SKILL_CATEGORY_RE.fullmatch(labelled.group(1).strip())
                        or _SKILL_CATEGORY_PREFIX_RE.fullmatch(labelled.group(1).strip())
                    )
                )
                or (prefixed_category and prefixed_category.end() < len(line))
            )
            if is_category_row:
                if pending_category_line:
                    rebuilt_lines.append(pending_category_line)
                pending_category_line = raw_line
                continue
            if pending_category_line and line:
                pending_category_line = f"{pending_category_line.rstrip()} {line}"
                continue
        if pending_category_line:
            rebuilt_lines.append(pending_category_line)
            pending_category_line = None
        rebuilt_lines.append(raw_line)
    if pending_category_line:
        rebuilt_lines.append(pending_category_line)

    in_skill_section = False
    pending_bullet = None
    pending_joined_lines = 0
    for raw_line in rebuilt_lines:
        line = raw_line.strip()
        header = _SKILL_SECTION_HEADER_RE.match(line)
        domain_sections = classify_combined_heading(line)
        domain_skill_header = (
            domain_sections == ["skills"]
            and not _re.search(r":\s*\S", line)
        )
        domain_boundary = classify_section_heading(line)
        if header or domain_skill_header:
            if pending_bullet:
                add_section_line(pending_bullet)
                pending_bullet = None
                pending_joined_lines = 0
            in_skill_section = True
            line = (header.group("content") or "").strip() if header else ""
            if not line:
                continue
        elif in_skill_section and (
            _NON_SKILL_SECTION_RE.match(line)
            or (domain_boundary and domain_boundary != "skills")
        ):
            if pending_bullet:
                add_section_line(pending_bullet)
                pending_bullet = None
                pending_joined_lines = 0
            in_skill_section = False
            continue
        if not in_skill_section:
            continue
        if not line:
            if pending_bullet:
                add_section_line(pending_bullet)
                pending_bullet = None
                pending_joined_lines = 0
            continue

        is_bullet = bool(_SKILL_BULLET_RE.match(line))
        if is_bullet:
            if pending_bullet:
                add_section_line(pending_bullet)
            pending_bullet = line
            pending_joined_lines = 0
            continue

        if pending_bullet:
            if _should_join_wrapped_skill(
                pending_bullet, line, pending_joined_lines
            ):
                pending_bullet = f"{pending_bullet} {line}"
                pending_joined_lines += 1
                continue
            add_section_line(pending_bullet)
            pending_bullet = None
            pending_joined_lines = 0
        add_section_line(line)

    if pending_bullet:
        add_section_line(pending_bullet)

    # A dedicated Skills section is authoritative for the displayed inventory.
    # Do not inflate it with words found in duties, publications, certificates,
    # employer names, or profile prose. Job matching remains a separate process.
    if extracted:
        return extracted

    # Add catalog-backed skills evidenced outside the explicit skill section.
    # Canonical IDs are used only for deduplication; display text always uses
    # the wording found in the resume.
    contextual_text = _contextual_skill_text(text)
    covered_catalog_skills = _catalog_skills_covered_by_explicit_items(
        extracted, vocabulary
    )
    for canonical, variants in vocabulary.items():
        if len(_re.sub(r'[^A-Za-z0-9]', '', canonical)) <= 2:
            continue
        if canonical.lower() in covered_catalog_skills:
            continue
        mention = _positive_skill_mention(contextual_text, variants)
        if mention and not any(
            mention.lower() in listed_skill.lower()
            or listed_skill.lower() in mention.lower()
            for listed_skill in extracted
        ):
            add(mention)

    return extracted


def _skill_variants(skill):
    normalized = skill.strip().lower()
    variants = {normalized}
    variants.update(SKILL_ALIASES.get(normalized, []))
    return variants


def _literal_skill_found(resume_text_lower, skill_variant):
    if _re.search(r'[^a-z0-9\s]', skill_variant):
        return skill_variant in resume_text_lower

    words = skill_variant.split()
    if len(words) > 1:
        pattern = r'\b' + r'\s+'.join(_re.escape(word) for word in words) + r'\b'
    else:
        pattern = r'\b' + _re.escape(skill_variant) + r'\b'

    return bool(_re.search(pattern, resume_text_lower))

def _fuzzy_skill_found(resume_text_lower, skill, threshold=85):
    """
    Fuzzy fallback for skill detection, used only after literal + alias matching
    fails. Handles inflection and word-order variation (e.g. 'planned lessons'
    for 'lesson planning') that exact matching misses, by comparing the skill
    phrase against sliding token windows of the resume.

    Limitations (kept deliberately so matches stay explainable):
    - Skipped for very short skills (< 4 chars, e.g. 'R', 'Go', 'C#') to avoid
      false positives inside unrelated words.
    - This is approximate string similarity, NOT semantic matching: synonyms
      that share no word stems (e.g. 'curriculum development' for 'lesson
      planning') still need an alias entry.
    - `threshold` (0-100) is tunable; raise it to be stricter, lower to be
      more lenient.
    """
    skill = skill.strip().lower()
    if not skill:
        return False
    # Short single-token skills stay literal-only to avoid spurious matches.
    if len(skill.split()) < 2 and len(skill) < 4:
        return False

    skill_tokens = skill.split()
    n = len(skill_tokens)
    resume_tokens = resume_text_lower.split()
    if not resume_tokens:
        return False

    best = 0
    for size in (n, n + 1, n + 2):
        if size <= 0 or size > len(resume_tokens):
            continue
        for i in range(0, len(resume_tokens) - size + 1):
            window = " ".join(resume_tokens[i:i + size])
            score = fuzz.token_set_ratio(skill, window)
            if score > best:
                best = score
                if best >= threshold:
                    return True
    return best >= threshold


def _stemmed_skill_found(resume_text_lower, skill, slack=3):
    """
    Stemmed-token fallback, used only after literal + alias + fuzzy matching
    fail. Collapses inflected forms so a skill still matches when the resume
    uses a different grammatical form or word order — e.g. 'lesson planning'
    matches 'planned engaging lessons', and 'classroom management' matches
    'managing the classroom' (shared stems {lesson, plan} / {classroom, manag}).

    Requires ALL stemmed skill tokens to co-occur inside a sliding window
    (skill length + `slack` words), so unrelated scattered mentions elsewhere
    in the resume do not produce a false match. Skipped for very short
    single-token skills (< 4 chars) to avoid noise, matching the fuzzy guard.
    """
    skill = skill.strip().lower()
    if not skill:
        return False
    if len(skill.split()) < 2 and len(skill) < 4:
        return False

    skill_set = set(_stem_tokens(skill))
    if not skill_set:
        return False
    resume_stems = _stem_tokens(resume_text_lower)
    if not resume_stems:
        return False

    win = len(skill_set) + slack
    if win >= len(resume_stems):
        return skill_set.issubset(set(resume_stems))
    for i in range(0, len(resume_stems) - win + 1):
        if skill_set.issubset(set(resume_stems[i:i + win])):
            return True
    return False


def _pick_phrase(options, seed, offset=0):
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    index = (int(digest[:8], 16) + offset) % len(options)
    return options[index]

def analyze_skills(resume_skills_text, required_skills_list):
    """
    Identifies matched and missing skills using word-boundary regex so that
    short skills like 'R', 'C', 'Go' are not falsely matched inside longer words
    (e.g. 'R' should not match 'recruiter', 'Go' should not match 'algorithm').
    """
    matched = []
    missing = []
    resume_text_lower = resume_skills_text.lower()

    for req_skill in required_skills_list:
        skill = req_skill.strip()
        if not skill:
            continue
        found = any(_literal_skill_found(resume_text_lower, variant) for variant in _skill_variants(skill))
        if not found:
            # Fall back to fuzzy matching so wording variations still match.
            found = _fuzzy_skill_found(resume_text_lower, skill)
        if not found:
            # Final fallback: stemmed-token match for inflected paraphrases.
            found = _stemmed_skill_found(resume_text_lower, skill)

        if found:
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing


def analyze_preferred_skills(resume_skills_text, preferred_skills_list):
    """
    Checks how many preferred (nice-to-have) skills appear in the resume.
    Returns (matched_preferred, total_preferred, bonus_pct) where bonus_pct
    is an additive bonus capped at 10.0 percentage points applied to the
    overall skill score.
    """
    if not preferred_skills_list:
        return [], 0, 0.0

    resume_text_lower = resume_skills_text.lower()
    matched_preferred = []

    for pref_skill in preferred_skills_list:
        skill = pref_skill.strip()
        if not skill:
            continue
        found = any(_literal_skill_found(resume_text_lower, variant) for variant in _skill_variants(skill))
        if not found:
            found = _fuzzy_skill_found(resume_text_lower, skill)
        if not found:
            found = _stemmed_skill_found(resume_text_lower, skill)
        if found:
            matched_preferred.append(skill)

    total_preferred = len([s for s in preferred_skills_list if s.strip()])
    if total_preferred == 0:
        return [], 0, 0.0

    # Bonus: up to 10 points scaled by how many preferred skills were found
    bonus_pct = round((len(matched_preferred) / total_preferred) * 10.0, 2)
    return matched_preferred, total_preferred, bonus_pct

def get_degree_rank(degree_str):
    import re as _re_rank
    if not degree_str:
        return 0
    degree_lower = degree_str.lower()

    # Use word-boundary patterns to avoid substring false positives
    # (e.g. "diploma" contains "ma", "ms", "ph" — plain substring checks misfire).
    def _matches(patterns):
        return any(_re_rank.search(p, degree_lower) for p in patterns)

    if _matches([r'\bph\.?d\b', r'\bdoctor']):
        return 5
    if _matches([
        r'\bmaster', r'\bm\.s\.?\b', r'\bm\.a\.?\b', r'\bmba\b',
        r'\b(?:med|maed|mpa|msw|mtech|msc)\b',
    ]):
        if _matches([
            r'\bongoing\b', r'\bincomplete\b', r'\b\d+\s+units?\b',
            r'complete(?:d)?\s+academic\s+requirements?',
        ]):
            return 3
        return 4
    if _matches([
        r'\bbachelor', r'\bb\.s\.?\b', r'\bb\.a\.?\b', r'\bb\.e\.?\b',
        r'\bcollege\s+graduate\b',
        r'\bbtech\b', r'\bbs\b', r'\bba\b', r'\bbe\b',
        r'\b(?:bsed|beed|bsie|bsn|bsit|bscs|bsa|bba|bshm|bsba|ab)\b',
        r'\b(?:bped|bece|bsece|bse|bsce|bpe|bsbio|bsmath|bseng)\b',
    ]):
        return 3
    if _matches([r'\bassociate']):
        return 2
    if _matches([r'\bhigh\s+school', r'\bdiploma\b', r'\bged\b']):
        return 1
    return 0

def _degree_label(rank):
    """Returns a human-readable degree label from a rank integer."""
    labels = {5: "Ph.D.", 4: "Master's degree", 3: "Bachelor's degree",
              2: "Associate's degree", 1: "High School diploma", 0: "unverified education"}
    return labels.get(rank, "unverified education")

def generate_analysis_narrative(job_title, fit_score, skill_score, exp_score, edu_score,
                                matched_skills, missing_skills, total_exp_years,
                                extracted_edu, experience_req, education_req,
                                disqualified_by_critical_skills,
                                matched_preferred=None, preferred_bonus=0.0,
                                matched_critical_skills=None, missing_critical_skills=None,
                                disqualification_reason=None):
    """
    Generates a unique, candidate-specific analysis narrative using all
    available evaluation signals. Each section is tailored to the actual data.
    """
    total_required = len(matched_skills) + len(missing_skills)
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)
    matched_critical_skills = matched_critical_skills or []
    missing_critical_skills = missing_critical_skills or []

    # --- Determine fit tier label ---
    if fit_score >= 75:
        tier = "strong"
    elif fit_score >= 50:
        tier = "moderate"
    else:
        tier = "weak"

    # --- Opening sentence: overall verdict ---
    if disqualification_reason:
        opening = (
            f"This candidate is marked Not Qualified for the {job_title} role because "
            f"{disqualification_reason}. The overall fit score remains {fit_score:.0f}% "
            f"to reflect the weighted skills, experience, and education evidence separately from the rule outcome."
        )
    elif disqualified_by_critical_skills:
        missing_critical_text = ', '.join(missing_critical_skills) or "one or more critical skills"
        opening = (
            f"This candidate has been automatically disqualified for the {job_title} role "
            f"because they are missing critical skill(s): {missing_critical_text}. "
            f"The overall fit score remains {fit_score:.0f}% to show the weighted evidence separately."
        )
    elif tier == "strong":
        opening = (
            f"This candidate is a strong match for the {job_title} role, "
            f"achieving an overall fit score of {fit_score:.0f}% - well above the qualification threshold."
        )
    elif tier == "moderate":
        opening = (
            f"This candidate shows a moderate fit for the {job_title} role "
            f"with an overall fit score of {fit_score:.0f}%, placing them in the 'For Review' category."
        )
    else:
        opening = (
            f"This candidate does not meet the minimum qualifications for the {job_title} role, "
            f"scoring {fit_score:.0f}% overall - below the required threshold."
        )

    # --- Skills section ---
    if total_required == 0:
        skills_text = "No specific required skills were defined for this role, so skill matching was not evaluated."
    elif matched_count == total_required:
        skills_text = (
            f"Notably, the candidate demonstrates all {total_required} required skill(s) "
            f"({', '.join(matched_skills)}), achieving a perfect skills match score."
        )
    elif matched_count == 0:
        skills_text = (
            f"The resume did not surface any of the {total_required} required skill(s) "
            f"({', '.join(missing_skills)}). This is the primary driver of the low overall score."
        )
    else:
        matched_str = ', '.join(matched_skills[:5]) + (f" and {matched_count - 5} more" if matched_count > 5 else "")
        missing_str = ', '.join(missing_skills[:5]) + (f" and {missing_count - 5} more" if missing_count > 5 else "")
        preferred_note = ""
        if matched_preferred:
            preferred_note = f" The candidate also matched {len(matched_preferred)} preferred skill(s) ({', '.join(matched_preferred[:3])}{'...' if len(matched_preferred) > 3 else ''}), earning a +{preferred_bonus:.0f}pt bonus."
        skills_text = (
            f"Out of {total_required} required skill(s), {matched_count} matched: {matched_str}. "
            f"The {missing_count} missing skill(s) - {missing_str} - account for the skills gap and "
            f"result in a required skills score of {skill_score:.0f}%.{preferred_note}"
        )

    if matched_critical_skills or missing_critical_skills:
        critical_parts = []
        if matched_critical_skills:
            critical_parts.append(f"matched critical skill(s): {', '.join(matched_critical_skills)}")
        if missing_critical_skills:
            critical_parts.append(f"missing critical skill(s): {', '.join(missing_critical_skills)}")
        skills_text = f"{skills_text} Critical-skill check: {'; '.join(critical_parts)}."

    # --- Experience section ---
    if experience_req and experience_req > 0:
        if total_exp_years >= experience_req:
            exp_text = (
                f"Their work history indicates approximately {total_exp_years:.1f} year(s) of relevant experience, "
                f"meeting or exceeding the {experience_req}-year requirement (experience score: {exp_score:.0f}%)."
            )
        elif total_exp_years > 0:
            exp_text = (
                f"Their work history reflects around {total_exp_years:.1f} year(s) of experience, "
                f"which falls short of the {experience_req}-year requirement "
                f"(experience score: {exp_score:.0f}%)."
            )
        else:
            exp_text = (
                f"No verifiable work experience timeline was detected in the resume. "
                f"The role requires {experience_req} year(s) of experience, resulting in an experience score of 0%."
            )
    else:
        if total_exp_years > 0:
            exp_text = (
                f"No minimum experience was specified for this role; the candidate's resume indicates "
                f"approximately {total_exp_years:.1f} year(s) of relevant work history."
            )
        else:
            exp_text = (
                "No minimum experience was specified for this role, and no explicit experience timeline "
                "was detected in the resume."
            )

    # --- Education section ---
    cand_edu_rank = 0
    if extracted_edu:
        cand_edu_rank = max(get_degree_rank(e['degree']) for e in extracted_edu)
    cand_edu_label = _degree_label(cand_edu_rank)
    job_edu_rank = get_degree_rank(education_req)
    job_edu_label = _degree_label(job_edu_rank) if job_edu_rank > 0 else None

    if not job_edu_label:
        edu_text = (
            f"No specific education requirement was set for this role. "
            f"The candidate's highest detected credential is a {cand_edu_label}."
        )
    elif cand_edu_rank >= job_edu_rank:
        edu_text = (
            f"The candidate's detected education level ({cand_edu_label}) meets or exceeds "
            f"the required {job_edu_label} for this role (education score: {edu_score:.0f}%)."
        )
    else:
        edu_text = (
            f"The candidate's detected education ({cand_edu_label}) does not meet "
            f"the required {job_edu_label} for this role (education score: {edu_score:.0f}%)."
        )

    # --- Closing recommendation ---
    if disqualification_reason or disqualified_by_critical_skills:
        closing = (
            "Recommendation: Do not advance unless the rule-triggering gap is resolved or verified differently "
            "during human review."
        )
    elif tier == "strong":
        closing = (
            "Recommendation: Advance to the next stage. This candidate presents a compelling profile "
            "and is well-suited for further evaluation."
        )
    elif tier == "moderate":
        closing = (
            "Recommendation: Human review advised. The candidate shows promise but has gaps that should "
            "be explored further during an interview or assessment."
        )
    else:
        closing = (
            "Recommendation: Not advised for advancement at this stage. The candidate's profile "
            "does not meet the minimum criteria required for this position."
        )

    return " ".join([opening, skills_text, exp_text, edu_text, closing])


def generate_decision_explanation(job_title, recommendation_label, fit_score,
                                  skill_score, exp_score, edu_score,
                                  text_similarity_score, matched_skills,
                                  missing_skills, matched_critical_skills,
                                  missing_critical_skills, matched_preferred,
                                  preferred_bonus, total_exp_years,
                                  experience_req, education_req):
    """
    Builds a concise explanation of the final screening decision.
    It keeps the key evidence, risks, scoring weights, and reviewer action visible.
    """
    matched_skills = matched_skills or []
    missing_skills = missing_skills or []
    matched_critical_skills = matched_critical_skills or []
    missing_critical_skills = missing_critical_skills or []
    matched_preferred = matched_preferred or []

    total_required = len(matched_skills) + len(missing_skills)
    matched_count = len(matched_skills)
    required_ratio = (matched_count / total_required) if total_required else 1.0
    seed = "|".join([
        job_title,
        recommendation_label,
        f"{fit_score:.2f}",
        ",".join(matched_skills),
        ",".join(missing_skills),
        f"{total_exp_years:.1f}",
    ])

    if recommendation_label == "Qualified":
        verdict_text = _pick_phrase([
            f"Decision: Qualified for {job_title}.",
            f"Decision: This applicant is qualified for {job_title}.",
            f"Decision: The resume reached Qualified status for {job_title}.",
        ], seed)
        verdict_text += f" Fit score: {fit_score:.0f}%. No hard disqualification rule was triggered."
    elif recommendation_label == "For Review":
        verdict_text = _pick_phrase([
            f"Decision: For Review for {job_title}.",
            f"Decision: This applicant needs manual review for {job_title}.",
            f"Decision: The resume is not a clear pass or fail for {job_title}.",
        ], seed)
        verdict_text += f" Fit score: {fit_score:.0f}%. Some useful evidence was found, but important gaps still need checking."
    else:
        verdict_text = _pick_phrase([
            f"Decision: Not Qualified for {job_title}.",
            f"Decision: This applicant is not qualified for {job_title}.",
            f"Decision: The resume did not meet the minimum screening standard for {job_title}.",
        ], seed)
        verdict_text += f" Fit score: {fit_score:.0f}%. One or more screening rules found a gap that is too large to ignore."

    strengths = []
    if matched_skills:
        strengths.append(f"required skill(s): {', '.join(matched_skills[:5])}")
    if matched_critical_skills:
        strengths.append(f"critical skill(s): {', '.join(matched_critical_skills[:5])}")
    if matched_preferred:
        strengths.append(f"preferred skill(s): {', '.join(matched_preferred[:5])}")
    if total_exp_years > 0:
        if experience_req and experience_req > 0:
            if total_exp_years >= experience_req:
                strengths.append(
                    f"work experience advantage: about {total_exp_years:.1f} year(s), meeting the {experience_req}-year requirement"
                )
            else:
                strengths.append(
                    f"work experience advantage: about {total_exp_years:.1f} year(s), but below the {experience_req}-year requirement"
                )
        else:
            strengths.append(f"work experience advantage: about {total_exp_years:.1f} year(s)")
    elif not experience_req or experience_req <= 0:
        strengths.append("no minimum experience requirement was set")
    if edu_score >= 100.0:
        strengths.append("education requirement appears satisfied")

    if strengths:
        strengths_text = _pick_phrase([
            f"Strengths: the applicant's main advantage is {strengths[0]}.",
            f"Strengths: the strongest evidence is {strengths[0]}.",
            f"Strengths: what works in the applicant's favor is {strengths[0]}.",
        ], seed, 1)
        if len(strengths) > 1:
            strengths_text += f" Also found: {'; '.join(strengths[1:])}."
    else:
        strengths_text = "Strengths: no clear advantage was detected from the configured requirements."

    weaknesses = []
    if missing_critical_skills:
        weaknesses.append(f"missing critical skill(s): {', '.join(missing_critical_skills)}")
    if missing_skills:
        weaknesses.append(f"missing required skill(s): {', '.join(missing_skills)}")
    if experience_req and experience_req > 0 and exp_score < 100.0:
        weaknesses.append(f"experience score did not fully clear the {experience_req}-year requirement")
    if edu_score < 100.0:
        weaknesses.append(f"education did not fully match the configured requirement ({education_req or 'not specified'})")

    if weaknesses:
        weaknesses_text = _pick_phrase([
            f"Concern: {weaknesses[0]}.",
            f"Gap: {weaknesses[0]}.",
            f"Weakness: {weaknesses[0]}.",
        ], seed, 2)
        if len(weaknesses) > 1:
            weaknesses_text += f" Also check: {'; '.join(weaknesses[1:])}."
    else:
        weaknesses_text = "Concern: no major gap was detected, but the extracted resume details should still be verified."

    risk_points = []

    if missing_critical_skills:
        risk_points.append(f"missing critical skill(s): {', '.join(missing_critical_skills)}")
    if total_required and len(matched_skills) == 0:
        risk_points.append("zero required skills were matched")
    elif total_required and required_ratio < 0.5:
        risk_points.append("less than half of the required skills were matched")
    if experience_req and experience_req > 0 and exp_score < 50.0:
        risk_points.append("experience is far below the minimum requirement")
    elif experience_req and experience_req > 0 and exp_score < 100.0:
        risk_points.append("experience is below the stated minimum requirement")
    if missing_skills and recommendation_label != "Not Qualified":
        risk_points.append("some required skills are still missing")

    risk_text = (
        _pick_phrase([
            f"Risk to review: {', '.join(risk_points)}.",
            f"Manual review should focus on: {', '.join(risk_points)}.",
            f"Most important risk: {', '.join(risk_points)}.",
        ], seed, 3)
        if risk_points else
        _pick_phrase([
            "Risk to review: no major rule-based risk was detected.",
            "Manual review focus: verify that the extracted resume details are accurate.",
            "Most important risk: no high-risk issue was detected by the configured rules.",
        ], seed, 4)
    )

    if total_required:
        requirements_text = (
            f"Matched {matched_count} out of {total_required} required skill(s). "
            f"Scores: skills {skill_score:.0f}%, experience {exp_score:.0f}%, education {edu_score:.0f}%."
        )
    else:
        requirements_text = (
            f"No required-skill list was provided. "
            f"Scores: experience {exp_score:.0f}%, education {edu_score:.0f}%."
        )

    preferred_text = (
        f"Preferred skill bonus: +{preferred_bonus:.2f}; it does not replace required skills."
        if matched_preferred else
        "No preferred-skill bonus was applied."
    )
    similarity_text = (
        f"Text similarity: {text_similarity_score:.0f}% reference metric only."
        if text_similarity_score is not None else
        "Text similarity was not available and did not affect the decision."
    )
    scoring_text = (
        _pick_phrase([
            "Scoring: skills 50%, experience 30%, education 20%.",
            "Score basis: skills 50%, experience 30%, education 20%.",
            "Fit score weights: skills 50%, experience 30%, education 20%.",
        ], seed, 5)
        + f" {requirements_text} {preferred_text} {similarity_text}"
    )

    if recommendation_label == "Qualified":
        reviewer_text = _pick_phrase([
            "Reviewer recommendation: advance to the next screening stage after verifying the extracted evidence.",
            "Reviewer recommendation: consider moving forward, but confirm the resume details manually.",
            "Reviewer recommendation: suitable for next-stage screening, pending human verification.",
        ], seed, 6)
    elif recommendation_label == "For Review":
        reviewer_text = _pick_phrase([
            "Reviewer recommendation: review the gaps before deciding whether to move forward.",
            "Reviewer recommendation: check the weaker areas manually before advancing.",
            "Reviewer recommendation: use this as a review queue item, not a final decision.",
        ], seed, 7)
    else:
        reviewer_text = _pick_phrase([
            "Reviewer recommendation: do not advance unless clearer evidence addresses the missing requirements.",
            "Reviewer recommendation: hold from advancement unless additional information resolves the gaps.",
            "Reviewer recommendation: current resume evidence is not strong enough for next-stage screening.",
        ], seed, 8)

    return "\n\n".join([
        verdict_text,
        strengths_text,
        weaknesses_text,
        scoring_text,
        f"{risk_text} {reviewer_text}",
    ])


def estimate_decision_confidence(recommendation_label, resume_text, contact_info,
                                 extracted_edu, extracted_exp, total_exp_years,
                                 required_skills, matched_skills, missing_skills,
                                 matched_critical_skills, missing_critical_skills,
                                 skill_score, exp_score, edu_score, fit_score,
                                 experience_req=0, education_req=None,
                                 min_fit_score=50.0):
    """
    Estimates how much trust the system should place in its own screening result.
    This is not a second decision; it tells the reviewer whether the extracted
    evidence is strong enough to trust or should be checked manually.
    """
    resume_text = resume_text or ""
    contact_info = contact_info or {}
    extracted_edu = extracted_edu or []
    extracted_exp = extracted_exp or []
    required_skills = [skill for skill in (required_skills or []) if str(skill).strip()]
    matched_skills = matched_skills or []
    missing_skills = missing_skills or []
    matched_critical_skills = matched_critical_skills or []
    missing_critical_skills = missing_critical_skills or []

    # Confidence is built from independently observable evidence.  Starting at
    # zero prevents a sparse or failed extraction from looking trustworthy just
    # because the resulting fit score is far from a decision threshold.
    score = 0
    strengths = []
    concerns = []
    text_length = len(resume_text.strip())
    words = _re.findall(r"[A-Za-z][A-Za-z'-]+", resume_text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0.0
    nonempty_lines = sum(bool(line.strip()) for line in resume_text.splitlines())

    # Length alone is easy to inflate (for example, repeated boilerplate), so
    # lexical variety and line structure also contribute to extraction quality.
    if text_length >= 700 and len(words) >= 100 and unique_ratio >= 0.18 and nonempty_lines >= 8:
        score += 25
        strengths.append("resume text is substantial and structurally readable")
    elif text_length >= 300 and len(words) >= 45 and unique_ratio >= 0.12:
        score += 16
        strengths.append("resume text is readable but has limited extraction coverage")
    elif text_length > 0:
        score += 4
        concerns.append("resume text is sparse or repetitive, so extraction may be incomplete")
    else:
        concerns.append("no readable resume text was available")

    candidate_name = (contact_info.get("name") or "").strip().lower()
    if candidate_name and candidate_name not in {"unknown candidate", "unknown", "n/a"}:
        score += 7
        strengths.append("candidate identity was detected")
    else:
        concerns.append("candidate name was not confidently detected")

    if contact_info.get("email") or contact_info.get("phone"):
        score += 3
        strengths.append("contact evidence was detected")

    total_required = len(required_skills)
    matched_required = len(matched_skills)
    required_ratio = matched_required / total_required if total_required else 1.0
    if total_required:
        score += 15
        if matched_required + len(missing_skills) == total_required:
            score += 5
            strengths.append("every configured required skill was evaluated")
        else:
            concerns.append("required-skill results do not reconcile with the configured list")

        # A missing-skill conclusion is only strong when the source extraction
        # itself is healthy; absence in sparse text is not reliable evidence.
        if required_ratio == 1.0:
            score += 5
            strengths.append("all required skills have direct text evidence")
        elif required_ratio == 0.0 and score < 40:
            concerns.append("missing skills may reflect incomplete text extraction")
    else:
        concerns.append("no required-skill list was configured")

    if missing_critical_skills:
        if text_length >= 300 and len(words) >= 45:
            score += 5
            strengths.append("critical-skill rule was evaluated against readable text")
        else:
            concerns.append("critical skills appear missing from incomplete resume text")
    elif matched_critical_skills:
        score += 5
        strengths.append("critical-skill evidence was found")

    if experience_req and experience_req > 0:
        if extracted_exp or total_exp_years > 0:
            score += 10
            strengths.append("work experience evidence was detected")
        else:
            concerns.append("work experience requirement exists but no work history was extracted")

        complete_exp = sum(
            bool(record.get("job_title")) and bool(record.get("company"))
            for record in extracted_exp if isinstance(record, dict)
        )
        if complete_exp:
            score += min(8, complete_exp * 3)
        elif extracted_exp:
            concerns.append("work-history records are missing a title or employer")

        record_years = sum(max(0, float(record.get("years") or 0)) for record in extracted_exp if isinstance(record, dict))
        if record_years and total_exp_years and abs(record_years - total_exp_years) > max(2.0, total_exp_years * 0.5):
            score -= 8
            concerns.append("total experience conflicts with extracted work-history durations")
        elif record_years and total_exp_years:
            score += 4
            strengths.append("experience totals agree with extracted work history")

        if 45.0 <= exp_score <= 55.0 or 90.0 <= exp_score < 100.0:
            score -= 5
            concerns.append("experience score is near a review boundary")
    elif extracted_exp or total_exp_years > 0:
        strengths.append("work history was detected")

    if education_req and get_degree_rank(education_req) > 0:
        if extracted_edu or edu_score >= 100.0:
            score += 8
            strengths.append("education evidence was detected")
        else:
            concerns.append("education requirement exists but no education history was extracted")

        complete_edu = any(
            isinstance(record, dict) and record.get("degree") and record.get("institution")
            for record in extracted_edu
        )
        if complete_edu:
            score += 4
        elif extracted_edu:
            concerns.append("education record is missing a degree or institution")

        if 80.0 <= edu_score < 100.0:
            score -= 4
            concerns.append("education score is close but not a full match")
    elif extracted_edu:
        strengths.append("education history was detected")

    threshold_distance = min(abs(fit_score - 75.0), abs(fit_score - min_fit_score))
    if threshold_distance <= 3.0:
        concerns.append("fit score is close to a decision threshold")
    elif threshold_distance >= 12.0:
        score += 10
        strengths.append("fit score is well separated from decision thresholds")
    else:
        score += 5

    if recommendation_label == "For Review":
        concerns.append("decision is intentionally routed for human review")

    score = max(0, min(100, round(score)))

    if score >= 75:
        level = "High"
    elif score >= 45:
        level = "Medium"
    else:
        level = "Low"

    if recommendation_label == "For Review" and level == "High":
        level = "Medium"

    reason_parts = [f"Evidence confidence score: {score}/100."]
    if strengths:
        reason_parts.append("Confidence support: " + "; ".join(strengths[:3]) + ".")
    if concerns:
        reason_parts.append("Reviewer should verify: " + "; ".join(concerns[:3]) + ".")
    if not reason_parts:
        reason_parts.append("Confidence is based on the available screening signals.")

    return level, " ".join(reason_parts)


def evaluate_candidate(resume_text, job_desc_text, required_skills, min_fit_score=50.0,
                       experience_req=0, education_req=None, requires_all_critical=False,
                       job_title="the target role", preferred_skills=None,
                       critical_skills=None):
    """
    Orchestrates the evaluation of a candidate's resume against a job description.
    Returns a dictionary with scores and recommendation.
    """
    from .nlp_pipeline import (
        clean_text, extract_contact_info, extract_education, 
        extract_years_of_experience, extract_experience_records, extract_certifications
    )
    from .matching_engine import calculate_fit_score, calculate_text_similarity
    
    # 1. Clean Texts
    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_desc_text)

    # 2. Required skill matching — run against original resume text (not cleaned) so that
    #    versioned skills like 'CSS3', 'ES6', 'HTML5', 'Python 3' are not stripped of digits.
    matched_skills, missing_skills = analyze_skills(resume_text, required_skills)
    matched_critical_skills, missing_critical_skills = analyze_skills(
        resume_text, critical_skills or []
    )

    # 2b. Preferred skill bonus — also uses original text for the same reason
    matched_preferred, total_preferred, preferred_bonus = analyze_preferred_skills(
        resume_text, preferred_skills or []
    )
    extracted_skills = extract_resume_skills(
        resume_text,
        [*(required_skills or []), *(critical_skills or []), *(preferred_skills or [])],
    )

    # 3. Extract contact info, education, experience
    contact_info = extract_contact_info(resume_text)
    extracted_edu = extract_education(resume_text)
    extracted_exp = extract_experience_records(resume_text)
    extracted_certifications = extract_certifications(resume_text)
    total_exp_years = extract_years_of_experience(resume_text)

    # 4. Required Skills Match Score (0–100), then apply preferred bonus (capped at 100)
    if required_skills:
        required_skill_score = (len(matched_skills) / len(required_skills)) * 100.0
    else:
        required_skill_score = 100.0
    skill_score = min(required_skill_score + preferred_bonus, 100.0)

    # 5. Experience Score with soft ceiling — extra years beyond 2× the requirement
    #    give diminishing returns rather than capping hard at 100%
    if not experience_req or experience_req <= 0:
        exp_score = 100.0
    else:
        raw_ratio = total_exp_years / experience_req
        if raw_ratio >= 1.0:
            # Met requirement — scale from 100% up, but cap at 100
            exp_score = 100.0
        else:
            # Partial credit; scale linearly up to 100%
            exp_score = round(raw_ratio * 100.0, 2)

    # 6. Education Score
    cand_rank = 0
    if extracted_edu:
        cand_rank = max(get_degree_rank(e['degree']) for e in extracted_edu)
    else:
        # Fallback: scan raw text directly
        cand_rank = get_degree_rank(resume_text)

    job_rank = get_degree_rank(education_req)
    if job_rank <= 0:
        edu_score = 100.0
    else:
        edu_score = min((cand_rank / job_rank) * 100.0, 100.0)

    # 7. Final Weighted Fit Score (4-component model:
    #    Skills 50%, Experience 25%, Education 15%, TF-IDF cosine similarity 10%).
    #    Cosine is weighted modestly because raw resume-vs-job cosine is
    #    structurally low; a higher weight would cap strong candidates' scores.
    text_similarity_score = calculate_text_similarity(cleaned_resume, cleaned_job)
    fit_score = calculate_fit_score(skill_score, exp_score, edu_score, text_similarity_score)
    
    # 9. Recommendation & Critical Skill Enforcement
    label = generate_recommendation(fit_score, min_fit_score)
    disqualified_by_critical_skills = False
    disqualification_reason = None

    if requires_all_critical and missing_critical_skills:
        label = "Not Qualified"
        disqualified_by_critical_skills = True
        missing_critical_text = ', '.join(missing_critical_skills)
        disqualification_reason = f"the resume is missing required critical skill(s): {missing_critical_text}"
    elif required_skills and len(matched_skills) == 0:
        label = "Not Qualified"
        disqualification_reason = "none of the configured required skills were found in the resume"
    elif experience_req and experience_req > 0 and exp_score < 50.0:
        label = "Not Qualified"
        disqualification_reason = "the detected work experience is far below the configured minimum requirement"
    elif experience_req and experience_req > 0 and exp_score < 100.0 and label == "Qualified":
        label = "For Review"

    # 9. Generate Unique Candidate-Specific Narrative
    summary = generate_analysis_narrative(
        job_title=job_title,
        fit_score=fit_score,
        skill_score=skill_score,
        exp_score=exp_score,
        edu_score=edu_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        total_exp_years=total_exp_years,
        extracted_edu=extracted_edu,
        experience_req=experience_req,
        education_req=education_req,
        disqualified_by_critical_skills=disqualified_by_critical_skills,
        matched_preferred=matched_preferred,
        preferred_bonus=preferred_bonus,
        matched_critical_skills=matched_critical_skills,
        missing_critical_skills=missing_critical_skills,
        disqualification_reason=disqualification_reason
    )
    decision_explanation = generate_decision_explanation(
        job_title=job_title,
        recommendation_label=label,
        fit_score=fit_score,
        skill_score=skill_score,
        exp_score=exp_score,
        edu_score=edu_score,
        text_similarity_score=text_similarity_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        matched_critical_skills=matched_critical_skills,
        missing_critical_skills=missing_critical_skills,
        matched_preferred=matched_preferred,
        preferred_bonus=preferred_bonus,
        total_exp_years=total_exp_years,
        experience_req=experience_req,
        education_req=education_req
    )
    confidence_level, confidence_reason = estimate_decision_confidence(
        recommendation_label=label,
        resume_text=resume_text,
        contact_info=contact_info,
        extracted_edu=extracted_edu,
        extracted_exp=extracted_exp,
        total_exp_years=total_exp_years,
        required_skills=required_skills,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        matched_critical_skills=matched_critical_skills,
        missing_critical_skills=missing_critical_skills,
        skill_score=skill_score,
        exp_score=exp_score,
        edu_score=edu_score,
        fit_score=fit_score,
        experience_req=experience_req,
        education_req=education_req,
        min_fit_score=min_fit_score,
    )
    return {
        "skill_score": round(skill_score, 2),
        "experience_score": round(exp_score, 2),
        "education_score": round(edu_score, 2),
        "text_similarity_score": text_similarity_score,
        "fit_score": round(fit_score, 2),
        "recommendation_label": label,
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_critical_skills": matched_critical_skills,
        "missing_critical_skills": missing_critical_skills,
        "matched_preferred": matched_preferred,
        "extracted_skills": extracted_skills,
        "summary": summary,
        "decision_explanation": decision_explanation,
        "contact_info": contact_info,
        "extracted_edu": extracted_edu,
        "extracted_exp": extracted_exp,
        "extracted_certifications": extracted_certifications,
        "total_exp_years": total_exp_years,
        # Per-component breakdown to make validation easy: shows each raw
        # sub-score, its weight, and how many points it contributed to the
        # final fit score. The weakest contributor is the bottleneck to fix.
        "score_breakdown": {
            "weights": {"skills": 0.50, "experience": 0.25,
                        "education": 0.15, "text_similarity": 0.10},
            "contributions": {
                "skills": round(skill_score * 0.50, 2),
                "experience": round(exp_score * 0.25, 2),
                "education": round(edu_score * 0.15, 2),
                "text_similarity": round(text_similarity_score * 0.10, 2),
            },
            "fit_score": round(fit_score, 2),
        }
    }
