import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from datetime import datetime

# Load models
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

EMAIL_RE = re.compile(r'(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])')
PHONE_RE = re.compile(
    r'(?<!\d)(?:'
    r'(?:\+?63|0)[- .]?(?:9\d{2}|2)[- .]?\d{3}[- .]?\d{4}'
    r'|(?:\+?\d{1,3}[- .]?)?\(?\d{2,4}\)?[- .]?\d{3}[- .]?\d{4}'
    r')(?!\d)'
)

RESUME_SECTION_RE = re.compile(
    r'^\s*(objective|summary|profile|professional\s+summary|technical\s+skills|skills|'
    r'education|work\s+experience|professional\s+experience|teaching\s+experience|'
    r'academic\s+experience|faculty\s+experience|experience|projects?|'
    r'certifications?|responsibilities|environment|languages|tools)\s*:?\s*$',
    re.IGNORECASE
)

TECH_OR_ROLE_RE = re.compile(
    r'\b(java|j2ee|jee|javascript|python|sql|html|css|spring|hibernate|angular|react|'
    r'node|aws|azure|docker|kubernetes|developer|engineer|architect|analyst|consultant|'
    r'programmer|manager|lead|administrator|full\s+stack|backend|frontend|software|'
    r'teacher|instructor|professor|lecturer|principal|dean|faculty|tutor|counselor|'
    r'accountant|auditor|cashier|recruiter|nurse|doctor|dentist|pharmacist|therapist|'
    r'caregiver|sales|marketing|representative|agent|receptionist|secretary|clerk|'
    r'assistant|technician|operator|mechanic|electrician|driver|chef|cook|server|'
    r'writer|designer|coordinator|supervisor|director|officer|executive|intern|'
    r'project\s+management|scrum\s+master|customer\s+service)\b',
    re.IGNORECASE
)

NAME_NOISE_RE = re.compile(
    r'\b(email|phone|mobile|contact|address|location|linkedin|github|portfolio|'
    r'curriculum\s+vitae|resume|application|university|college|school|institute|'
    r'corporation|company|inc\.?|ltd\.?|solutions?|services?|systems?|technologies|'
    r'enterprises?|industries|agency|group|center|centre|foundation|association|'
    r'street|road|avenue|city|province|country|'
    r'bachelor|master|diploma|objective|summary|profile)\b',
    re.IGNORECASE
)

NAME_PARTICLES = {'de', 'del', 'di', 'da', 'dos', 'das', 'van', 'von', 'la', 'le', 'du', 'bin'}
NAME_PREFIXES = {'mr', 'mrs', 'ms', 'miss', 'dr', 'engr', 'prof', 'atty'}
NAME_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
NAME_CREDENTIALS = {'cpa', 'rn', 'lpt', 'rpm', 'cphr', 'pmp', 'phd', 'md', 'mba', 'csc'}
EMAIL_NAME_NOISE = {
    'admin', 'applicant', 'application', 'career', 'contact', 'cv', 'email',
    'hello', 'hr', 'info', 'jobs', 'mail', 'office', 'recruitment', 'resume',
}

def clean_text(text):
    """
    Cleans raw text by removing stopwords, punctuation, and extra whitespace.
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits (optional, but good for pure text/skill matching)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    cleaned_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return " ".join(cleaned_tokens)

def extract_entities(text):
    """
    Extracts named entities from text using spaCy.
    Returns a dictionary of organizations, locations, and other relevant entities.
    """
    doc = nlp(text)
    
    entities = {
        'ORG': set(),
        'GPE': set(),
        'PERSON': set(),
        'NOUN_CHUNKS': set()
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].add(ent.text)
            
    # Extract noun chunks as potential skills or keyphrases
    for chunk in doc.noun_chunks:
        cleaned_chunk = chunk.text.lower().strip()
        if len(cleaned_chunk.split()) <= 3: # Keep short phrases
            entities['NOUN_CHUNKS'].add(cleaned_chunk)
            
    # Convert sets to lists for JSON serialization later
    return {k: list(v) for k, v in entities.items()}

def extract_skills(text, predefined_skills=None):
    """
    Extracts skills by matching against a predefined list of skills (e.g., from a job description).
    """
    if not text or not predefined_skills:
        return []
        
    text_lower = text.lower()
    extracted = []
    
    for skill in predefined_skills:
        # Simple string matching; could be enhanced with regex for exact word boundaries
        if re.search(r'\b' + re.escape(skill.lower().strip()) + r'\b', text_lower):
            extracted.append(skill.strip())
            
    return extracted


def extract_certifications(text):
    """Extract professional certifications, licenses, and board-exam credentials."""
    if not text:
        return []

    month = (r'(?:January|February|March|April|May|June|July|August|September|'
             r'October|November|December)')
    date_re = re.compile(rf'\b(?:{month}\s*)?(?:19|20)\d{{2}}\b', re.IGNORECASE)
    lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines() if line.strip()]
    records = []

    def add(name, credential_type='Certification', date=None, issuer=None):
        name = re.sub(r'\s+', ' ', name or '').strip(' .,:;|-')
        if not name:
            return
        key = re.sub(r'[^a-z0-9]+', '', name.lower())
        aliases = {
            'lpt': 'licensedprofessionalteacher',
            'letpasser': 'licensedprofessionalteacher',
            'licensedprofessionalteacherletpasser': 'licensedprofessionalteacher',
            'rpm': 'registeredpsychometrician',
        }
        key = aliases.get(key, key)
        for existing in records:
            existing_key = re.sub(r'[^a-z0-9]+', '', existing['certification_name'].lower())
            existing_key = aliases.get(existing_key, existing_key)
            if existing_key == key:
                if date and not existing['date_obtained']:
                    existing['date_obtained'] = date
                if issuer and not existing['issuer']:
                    existing['issuer'] = issuer
                return
        records.append({
            'certification_name': name,
            'credential_type': credential_type,
            'issuer': issuer,
            'date_obtained': date,
        })

    joined = '\n'.join(lines)

    # Common regulated Philippine credentials and their resume abbreviations.
    known = [
        (r'\bRegistered\s+Psychometrician\b|\bRPm\b', 'Registered Psychometrician', 'Professional License'),
        (r'\bLicensed\s+Professional\s+Teacher\b|\bLET\s*Passer\b|\bLPT\b', 'Licensed Professional Teacher', 'Professional License'),
        (r'\bCertified\s+Professional\s+in\s+Human\s+Resources\b|\bCPHR\b', 'Certified Professional in Human Resources (CPHR)', 'Certification'),
        (r'\bRegistered\s+Nurse\b|\bRN\b', 'Registered Nurse', 'Professional License'),
        (r'\bCertified\s+Public\s+Accountant\b|\bCPA\b', 'Certified Public Accountant', 'Professional License'),
        (r'\bRegistered\s+Criminologist\b|\bRCRIM\b', 'Registered Criminologist', 'Professional License'),
        (r'\bCivil\s+Service\s+(?:Professional\s+)?Eligible\b|\bCSE\s+Passer\b', 'Civil Service Eligibility', 'Eligibility'),
    ]
    for pattern, canonical, kind in known:
        for index, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                date_match = date_re.search(line)
                if not date_match and index + 1 < len(lines):
                    next_date = date_re.fullmatch(lines[index + 1].strip(' ()'))
                    date_match = next_date
                add(canonical, kind, date_match.group(0) if date_match else None)

    # Preserve explicitly named certifications not covered by the aliases above.
    section_active = False
    for index, line in enumerate(lines):
        if re.fullmatch(r'(?:professional\s+)?(?:certifications?|licenses?|licensure|eligibility)', line, re.IGNORECASE):
            section_active = True
            continue
        if section_active and RESUME_SECTION_RE.match(line):
            section_active = False
        explicit = re.search(
            r'(?i)\b((?:certified|certification\s+(?:in|on)|licensed|registered)\s+'
            r'[A-Za-z][A-Za-z0-9 &/+.#-]{2,80})', line
        )
        if explicit or (section_active and len(line) <= 120):
            candidate = explicit.group(1) if explicit else line
            date_match = date_re.search(candidate)
            candidate = date_re.sub('', candidate).strip(' .,:;|-()')
            if not re.fullmatch(r'(?i)(certifications?|licenses?|licensure|eligibility)', candidate):
                add(candidate, 'Professional License' if re.search(r'(?i)licensed|registered|board|passer', candidate) else 'Certification', date_match.group(0) if date_match else None)

    return records

def extract_contact_info(text):
    """
    Extracts name, email, and phone number from resume text. Candidate names
    are ranked from header evidence instead of accepting the first short line.
    """
    name = None
    email = None
    phone = None

    def _clean_name_candidate(value):
        value = EMAIL_RE.sub('', value or '')
        value = PHONE_RE.sub('', value)
        value = re.sub(r'https?://\S+|www\.\S+', '', value, flags=re.IGNORECASE)
        value = re.sub(r'\b(linkedin|github|portfolio)\s*:?\s*\S*', '', value, flags=re.IGNORECASE)
        value = ''.join(
            char if char.isalpha() or char in " .'- ," else ' '
            for char in value
        )
        value = re.sub(r'\s+', ' ', value).strip(' .-')

        # Remove honorifics and professional credentials, but preserve family
        # suffixes such as Jr. and III.
        words = value.split()
        if words and words[0].lower().rstrip('.') in NAME_PREFIXES:
            value = ' '.join(words[1:])
        comma_parts = [part.strip() for part in value.split(',')]
        while len(comma_parts) > 1 and comma_parts[-1].lower().rstrip('.') in NAME_CREDENTIALS:
            comma_parts.pop()
        if len(comma_parts) == 2:
            right_token = comma_parts[1].lower().rstrip('.')
            if right_token in NAME_CREDENTIALS:
                value = comma_parts[0]
            elif comma_parts[0] and comma_parts[1]:
                # Common directory format: "Surname, Given Middle".
                value = f"{comma_parts[1]} {comma_parts[0]}"
        elif comma_parts:
            value = ' '.join(comma_parts)
        value = value.replace(',', ' ')
        words = value.split()
        while words and words[-1].lower().rstrip('.') in NAME_CREDENTIALS:
            words.pop()
        value = ' '.join(words)
        value = re.sub(r'\s+', ' ', value).strip(' .-')
        return value

    def _format_person_name(value):
        def _capitalize_name_word(word):
            hyphenated = []
            for hyphen_part in word.split('-'):
                apostrophe_parts = hyphen_part.split("'")
                hyphenated.append("'".join(
                    part[:1].upper() + part[1:].lower() if part else part
                    for part in apostrophe_parts
                ))
            return '-'.join(hyphenated)

        formatted = []
        for index, word in enumerate(value.split()):
            lower = word.lower().rstrip('.')
            if lower in NAME_PARTICLES and index > 0:
                formatted.append(lower)
            elif lower in NAME_SUFFIXES:
                formatted.append(lower.upper() if lower in {'ii', 'iii', 'iv', 'v'} else lower.capitalize() + '.')
            elif len(word.rstrip('.')) == 1:
                formatted.append(word[0].upper() + ('.' if word.endswith('.') else ''))
            else:
                formatted.append(_capitalize_name_word(word))
        return ' '.join(formatted)

    def _is_probable_person_name(value, allow_single=False):
        candidate = _clean_name_candidate(value)
        if not candidate:
            return False
        if len(candidate) > 60 or NAME_NOISE_RE.search(candidate):
            return False
        if RESUME_SECTION_RE.match(candidate):
            return False
        if re.match(r'(?i)^(location|school|institution|university|college|company|client|employer|role|position|designation|title)\b', candidate):
            return False
        words = candidate.split()
        if len(words) > 6:
            return False
        if TECH_OR_ROLE_RE.search(candidate):
            return False
        if len(words) == 1:
            return allow_single and len(words[0]) >= 3 and words[0].isalpha()
        for word in words:
            plain = word.strip(".'-")
            if not plain or not all(char.isalpha() or char in ".'-" for char in word):
                return False
            lower_plain = plain.lower()
            if lower_plain in NAME_CREDENTIALS:
                return False
            if len(plain) == 1 or lower_plain in NAME_PARTICLES or lower_plain in NAME_SUFFIXES:
                continue
            if len(plain) < 2:
                return False
        return True

    def _candidate_variants(line_text):
        variants = [line_text.strip()]

        labelled = re.match(
            r'^\s*(?:(?:candidate|applicant)\s+)?(?:full\s+)?name\s*[:\-]\s*(.+)$',
            line_text,
            re.IGNORECASE,
        )
        if labelled:
            variants.insert(0, labelled.group(1).strip())

        cv_of = re.match(
            r'^\s*(?:curriculum\s+vitae|resume)\s+(?:of|for)\s+(.+)$',
            line_text,
            re.IGNORECASE,
        )
        if cv_of:
            variants.insert(0, cv_of.group(1).strip())

        variants.extend(
            part.strip()
            for part in re.split(r'\s+\|\s+|\t+|\s{2,}|\s+[/-]\s+', line_text)
            if part.strip()
        )

        unique = []
        seen = set()
        for variant in variants:
            key = variant.lower()
            if key not in seen:
                seen.add(key)
                unique.append(variant)
        return unique
    
    # 1. Extract Email
    email_match = EMAIL_RE.search(text)
    if email_match:
        email = email_match.group(0).strip(' .,;:').lower()
        
    # 2. Extract Phone
    phone_match = PHONE_RE.search(text)
    if phone_match:
        phone = phone_match.group(0).strip()
        
    # 3. Extract Name
    # Prefer explicit candidate-name labels when present.
    name_label_match = re.search(
        r'(?im)^\s*(?:(?:candidate|applicant)\s+)?(?:full\s+)?name\s*[:\-]\s*([^\n\r]+)',
        text,
    )
    if name_label_match:
        for variant in _candidate_variants(name_label_match.group(1)):
            if _is_probable_person_name(variant, allow_single=True):
                name = _format_person_name(_clean_name_candidate(variant))
                break

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    header_lines = lines[:12]

    email_name = None
    email_identity_tokens = set()
    email_compact = ''
    if email:
        local_part = email.split('@', 1)[0].split('+', 1)[0]
        local_part = re.sub(r'\d+', ' ', local_part)
        local_part = re.sub(r'[._-]+', ' ', local_part).strip()
        local_tokens = [part for part in local_part.split() if len(part) > 1]
        email_identity_tokens = {
            part.lower() for part in local_tokens
            if (not TECH_OR_ROLE_RE.fullmatch(part)
                and part.lower() not in EMAIL_NAME_NOISE
                and part.lower() not in NAME_CREDENTIALS)
        }
        email_compact = ''.join(sorted(email_identity_tokens, key=lambda token: local_part.lower().find(token)))
        derived = ' '.join(
            part.capitalize() for part in local_tokens
            if part.lower() in email_identity_tokens
        )
        if len(derived.split()) >= 2 and _is_probable_person_name(derived):
            email_name = _format_person_name(derived)

    if not name:
        person_entities = set()
        blocked_entities = set()
        header_doc = nlp("\n".join(header_lines[:8]))
        for ent in header_doc.ents:
            cleaned_entity = _clean_name_candidate(ent.text).lower()
            if ent.label_ == 'PERSON':
                person_entities.add(cleaned_entity)
            elif ent.label_ in {'ORG', 'GPE', 'LOC', 'FAC'}:
                blocked_entities.add(cleaned_entity)

        ranked_candidates = [(110, email_name)] if email_name else []
        email_tokens = set(email_name.lower().split()) if email_name else set()
        for index, line in enumerate(header_lines):
            for variant_index, variant in enumerate(_candidate_variants(line)):
                cleaned = _clean_name_candidate(variant)
                cleaned_words = cleaned.split()
                allow_single = (
                    len(cleaned_words) == 1
                    and (
                        cleaned.lower() in email_identity_tokens
                        or any(
                            token.startswith(cleaned.lower())
                            for token in email_identity_tokens
                        )
                    )
                )
                if not _is_probable_person_name(cleaned, allow_single=allow_single):
                    continue
                words = cleaned.split()
                score = 100 - (index * 5) - variant_index
                if 2 <= len(words) <= 4:
                    score += 15
                if cleaned.isupper() or cleaned.istitle():
                    score += 8
                if cleaned.lower() in person_entities:
                    score += 20
                if cleaned.lower() in blocked_entities:
                    score -= 35
                candidate_tokens = set(cleaned.lower().split())
                if email_tokens and len(candidate_tokens & email_tokens) >= min(2, len(email_tokens)):
                    score += 30
                candidate_compact = ''.join(
                    char.lower() for char in cleaned if char.isalpha()
                )
                if email_compact and candidate_compact == email_compact:
                    score += 40
                if re.match(r'(?i)^\s*(?:curriculum\s+vitae|resume)\s+(?:of|for)\s+', line):
                    score += 25
                ranked_candidates.append((score, cleaned))

        if ranked_candidates:
            _, best_candidate = max(ranked_candidates, key=lambda item: item[0])
            name = _format_person_name(best_candidate)

    if not name and email_name:
        name = email_name

    if name and email_identity_tokens:
        name_words = name.split()
        merged_words = []
        index = 0
        while index < len(name_words):
            merged = None
            for end in range(min(len(name_words), index + 2), index + 1, -1):
                compact = ''.join(
                    char.lower()
                    for word in name_words[index:end]
                    for char in word
                    if char.isalpha()
                )
                segment_words = name_words[index:end]
                should_merge_split_token = (
                    len(segment_words) == 2
                    and len(segment_words[1].strip(".'-")) <= 3
                    and segment_words[0].lower().strip(".'-") not in NAME_PARTICLES
                )
                if compact in email_identity_tokens and should_merge_split_token:
                    merged = _format_person_name(compact)
                    index = end
                    break
            if merged:
                merged_words.append(merged)
            else:
                merged_words.append(name_words[index])
                index += 1
        name = _format_person_name(' '.join(merged_words))
            
    return {
        'name': name or 'Unknown Candidate',
        'email': email or 'Unknown Email',
        'phone': phone or 'Unknown Phone'
    }

def extract_education(text):
    """
    Extracts educational degrees and institutions from resume text.

    Strategy:
    1. Try to isolate the 'Education' section of the resume first so that
       lines like 'Bachelor's degree required' in job descriptions or skills
       sections don't create false positives.
    2. Within that section, use tighter regex patterns — short abbreviations
       (bs, ba, ms, ma) are only accepted when paired with recognisable
       contextual keywords (e.g. 'in', 'of', 'science', 'arts').
    3. If no Education section header is found, fall back to scanning the
       full text but with the same tighter patterns.
    4. Clean up institution names — strip leading/trailing punctuation and
       discard values that are clearly not school names.
    """

    # ------------------------------------------------------------------
    # Step 1: Isolate the Education section if one exists
    # ------------------------------------------------------------------
    EDU_SECTION_HEADERS = re.compile(
        r'^\s*(?:education(?:al)?\s*(?:background|history|qualification|details)?'
        r'|academic\s*(?:background|history|qualification)?'
        r'|qualifications?'
        r'|degrees?)\s*:?\s*(?P<content>.*)$',
        re.IGNORECASE
    )
    # Section headers that signal the end of the education section
    NEXT_SECTION_HEADERS = re.compile(
        r'^\s*(experience|professional\s+experience|employment|work\s*history|skills?'
        r'|technical\s+skills|certifications?|professional\s+(?:training|certifications?)'
        r'|professional\s+training\s+and\s+certifications?'
        r'|projects?|awards?|achievements?|interests?|references?|languages?|publications?'
        r'|summary|objective|profile|activities|affiliations?|seminars?(?:\s+and\s+trainings?)?'
        r'|personal\s+(?:information|data)|(?:(?:teaching|professional|work)\s+)?experiences?)\s*:?\s*$',
        re.IGNORECASE
    )

    lines = [line.strip() for line in text.split('\n')]
    edu_lines = []
    in_edu_section = False

    for line in lines:
        if not line:
            if in_edu_section:
                edu_lines.append('')   # preserve blank line as context
            continue
        header_match = EDU_SECTION_HEADERS.match(line)
        if not header_match and re.match(r'^education\b', line, re.IGNORECASE):
            # Two-column PDFs may append text from the other column to a
            # section heading (for example, "EDUCATION students, foster...").
            in_edu_section = True
            continue
        if header_match:
            in_edu_section = True
            inline_content = header_match.group('content').strip()
            if inline_content:
                edu_lines.append(inline_content)
            continue
        if in_edu_section and NEXT_SECTION_HEADERS.match(line):
            break
        if in_edu_section:
            edu_lines.append(line)

    # If we found a dedicated section use it; otherwise scan everything
    scan_lines = edu_lines if edu_lines else [l.strip() for l in lines if l.strip()]

    # ------------------------------------------------------------------
    # Step 2: Degree patterns — ordered most-specific first
    # ------------------------------------------------------------------
    DEGREE_PATTERNS = [
        # PhD
        (re.compile(
            r'\b(ph\.?d\.?|doctor\s+of\s+philosophy|doctorate)\b', re.I),
         "Ph.D."),

        # Master's — require the full word or unambiguous abbreviations
        (re.compile(
            r'\b(master(?:\'s)?\s+(?:of|in|degree)'
            r'|masters\s+degree'
            r'|m\.s\.?|m\.a\.?|m\.b\.a\.?|mba'
            r'|m\.?(?:ed|aed|pa|sw|tech|sc)\.?)\b',
            re.I),
         "Master's"),

        # Bachelor's — accept b.s./b.a./b.e./b.tech. as abbreviations,
        # but require standalone 'bs'/'ba'/'be' to be followed by context
        (re.compile(
            r'\b(bachelor(?:\'s)?(?:\s+of|\s+in|\s+degree)?'
            r'|b\.s\.?|b\.a\.?|b\.e\.?|b\.tech\.?|btech'
            r'|bsed|beed|bsn|bsit|bscs|bsa|bba|bshm|bsba|ab'
            r'|bachelor\s+of\s+(?:science|arts|engineering|technology|commerce|business)'
            r'|bs\s+in\b|ba\s+in\b|be\s+in\b)\b',
            re.I),
         "Bachelor's"),

        # Associate's
        (re.compile(
            r'\b(associate(?:\'s)?(?:\s+degree|\s+of|\s+in)?'
            r'|a\.s\.?|a\.a\.?)\b',
            re.I),
         "Associate's"),

        # High School
        (re.compile(
            r'\b(high\s+school(?:\s+diploma)?|secondary\s+school|ged|diploma)\b',
            re.I),
         "High School Diploma"),
    ]

    # ------------------------------------------------------------------
    # Step 3: Institution extraction helpers
    # ------------------------------------------------------------------
    INSTITUTION_RE = re.compile(
        r'((?:(?:[A-Z][A-Za-z&.\'-]*|of|the)\s+){0,7}'
        r'(?:University|College|Colegio|Institute|School|Academy|Polytechnic)'
        r'(?:\s+(?:[A-Z][A-Za-z&.\'-]*|of|the)){0,4})'
    )
    # Strings that are definitely not school names
    JUNK_INSTITUTION = re.compile(
        r'^\s*(unknown institution|bachelor|master|ph\.?d|associate'
        r'|diploma|degree|n/?a|none|required|preferred|group\s+project'
        r'|product\s+management|project\s+management|professional\s+experience'
        r'|training|certification|skills?)\s*$',
        re.IGNORECASE
    )

    INSTITUTION_NOISE_RE = re.compile(
        r'\b(project|product|client|role|manager|developer|engineer|scrum|'
        r'responsibilities|environment|training|certification|skills?|experience)\b',
        re.IGNORECASE
    )

    def _extract_institution(line_text, allow_org_fallback=True):
        """Try to pull a school name from a single line."""
        labelled = re.match(
            r'^\s*(?:school|institution|university|college|colegio|academy)\s*:\s*(.+)$',
            line_text,
            re.IGNORECASE
        )
        if labelled:
            name = re.sub(r'\b(?:19|20)\d{2}\b.*$', '', labelled.group(1)).strip(' .,;:|-')
            if not JUNK_INSTITUTION.match(name) and 2 < len(name) <= 100:
                return name
        colegio = re.search(r'(?i)\b(Colegio\s+de\s+[A-Za-z .\'-]+)', line_text)
        if colegio:
            return re.split(r'\t|\s{2,}', colegio.group(1))[0].strip(' .,;:|-')
        # Regex: look for University/College/Institute/etc.
        m = INSTITUTION_RE.search(line_text)
        if m:
            name = re.sub(r'\s+', ' ', m.group(1)).strip(' .,;:|-')
            if not JUNK_INSTITUTION.match(name) and len(name) > 4:
                return name
        if not allow_org_fallback or INSTITUTION_NOISE_RE.search(line_text):
            return None
        # spaCy ORG fallback
        doc = nlp(line_text)
        orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == 'ORG']
        for org in orgs:
            if (not JUNK_INSTITUTION.match(org)
                    and not INSTITUTION_NOISE_RE.search(org)
                    and len(org) > 3
                    and len(org.split()) <= 10):
                return org
        return None

    def _extract_degree_description(line_text, degree_match, fallback_label):
        """Preserve the credential text instead of reducing it to a level."""
        description = line_text[degree_match.start():].strip()
        description = re.split(r'\b(?:19|20)\d{2}\b', description, maxsplit=1)[0]
        description = re.split(
            r'\b(?:school|institution|university|college|colegio|academy)\s*:',
            description,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        school_after_separator = re.search(
            r'(?:[,;|]|\s+-\s+|\s+at\s+)\s*'
            r'(?=[A-Za-z&.\' -]{0,60}\b(?:University|College|Institute|School|Academy|Polytechnic)\b)',
            description,
            re.IGNORECASE,
        )
        if school_after_separator:
            description = description[:school_after_separator.start()]
        description = re.sub(r'\s+', ' ', description).strip(' .,;:|-')
        return (description or fallback_label)[:255]

    # ------------------------------------------------------------------
    # Step 4: Scan lines for degree matches
    # ------------------------------------------------------------------
    records = []
    for i, line in enumerate(scan_lines):
        if not line:
            continue
        matched_degree = None
        degree_match = None
        for pattern, label in DEGREE_PATTERNS:
            match = pattern.search(line)
            if match:
                matched_degree = label
                degree_match = match
                break

        if not matched_degree:
            continue
        if re.match(r'^\s*\([^)]*high\s+school[^)]*\)\s*$', line, re.IGNORECASE):
            continue

        # Try to find institution on the same line first
        institution = _extract_institution(line, allow_org_fallback=False)

        # If not found, inspect nearby lines in either direction. Education
        # layouts commonly place the school before or after the degree.
        if not institution:
            nearby_indices = [i - 1, i + 1, i - 2, i + 2]
            for j in nearby_indices:
                if 0 <= j < len(scan_lines) and scan_lines[j].strip():
                    institution = _extract_institution(scan_lines[j])
                if institution:
                    break

        records.append({
            'degree': _extract_degree_description(line, degree_match, matched_degree),
            'institution': institution or 'Unknown Institution'
        })

    # ------------------------------------------------------------------
    # Step 5: Deduplicate — prefer records that have a real institution
    # ------------------------------------------------------------------
    unique_records = []
    seen = set()
    for rec in records:
        key = (rec['degree'].lower(), rec['institution'].lower())
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(rec)

    return unique_records


def extract_years_of_experience(text):
    """
    Estimates total years of experience from resume text.
    """
    # 1. Prefer explicit total-experience statements. The word "experience"
    # is mandatory so education durations and age are not counted as work.
    explicit_patterns = [
        r'\b(?:over|more\s+than|around|approximately|about)?\s*'
        r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?(?:strong\s+)?'
        r'(?:software\s+|work\s+|professional\s+|relevant\s+|industry\s+|IT\s+)?experience\b',
        r'\bexperience\s+(?:of\s+|totaling\s+|totalling\s+|for\s+|over\s+)?'
        r'(\d+(?:\.\d+)?)\+?\s*years?\b',
        r'\b(\d+(?:\.\d+)?)\s*[- ]year\s+'
        r'(?:software\s+|work\s+|professional\s+|relevant\s+)?experience\b',
    ]
    exp_mentions = []
    for pattern in explicit_patterns:
        exp_mentions.extend(re.findall(pattern, text, re.IGNORECASE))
    if exp_mentions:
        try:
            realistic_mentions = [float(value) for value in exp_mentions if 0 < float(value) <= 50]
            if realistic_mentions:
                return max(realistic_mentions)
        except ValueError:
            pass

    # 2. Prefer experience records extracted from a realistic section structure.
    records = extract_experience_records(text)
    if records:
        # Calculate the union of dated employment periods so simultaneous
        # roles (for example, a full-time role plus freelance work) are not
        # double-counted as extra calendar experience.
        section_match = re.search(
            r'(?ims)^\s*(?:work\s+)?experiences?\s*:?\s*$\n(?P<body>.*?)(?=^\s*(?:education|skills?|certifications?|achievements?|leadership\s*(?:&|and)\s*activities|affiliations?|seminars?|references?)\s*:?\s*$|\Z)',
            text,
        )
        dated_text = section_match.group('body') if section_match else ''
        month_names = {
            name.lower(): number for number, names in enumerate((
                ('january', 'jan'), ('february', 'feb'), ('march', 'mar'),
                ('april', 'apr'), ('may',), ('june', 'jun'), ('july', 'jul'),
                ('august', 'aug'), ('september', 'sep'), ('october', 'oct'),
                ('november', 'nov'), ('december', 'dec')
            ), start=1) for name in names
        }
        token = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}|(?:19|20)\d{2}'
        range_re = re.compile(rf'(?P<start>{token})\s*(?:-|–|—|to)\s*(?P<end>Present|Current|{token})', re.I)

        def ordinal(value, is_end=False):
            value = value.strip().lower()
            if value in {'present', 'current'}:
                now = datetime.now()
                return now.year * 12 + now.month - 1
            named = re.match(r'([a-z]+)\s*(\d{4})', value)
            if named:
                return int(named.group(2)) * 12 + month_names[named.group(1)[:3]] - 1
            year = int(value)
            return year * 12

        intervals = []
        for match in range_re.finditer(dated_text):
            start = ordinal(match.group('start'))
            end = ordinal(match.group('end'), is_end=True)
            if end >= start and end - start <= 600:
                intervals.append((start, max(end, start + 1)))
        if intervals:
            intervals.sort()
            merged = [list(intervals[0])]
            for start, end in intervals[1:]:
                if start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            return min(round(sum(end - start for start, end in merged) / 12.0, 1), 40.0)

        total_from_records = sum(
            float(record.get('years', 0) or 0)
            for record in records
            if record.get('years') is not None
        )
        if total_from_records > 0:
            return min(round(total_from_records, 1), 40.0)

    # 3. Do not count arbitrary date ranges from education or certifications.
    # Patterns: "2018 - 2021", "2015 to Present", "2017-Present"
    current_year = datetime.now().year
    ranges = re.findall(r'\b(19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*(Present|19\d{2}|20\d{2})\b', text, re.IGNORECASE)
    ranges = []
    total_years = 0.0
    for start_str, end_str in ranges:
        try:
            start = int(start_str)
            end = current_year if end_str.lower() == 'present' else int(end_str)
            if end >= start and end - start <= 40:
                total_years += (end - start)
        except ValueError:
            pass

    if total_years > 0:
        return min(round(total_years, 1), 40.0)

    return 0.0

def extract_experience_records(text):
    """
    Extracts individual work experience records (title, company, years).

    Strategy:
    1. Isolate the Experience section if a section header is present so that
       bullet-point duty lines in other sections don't trigger false matches.
    2. Split the section into blocks (separated by blank lines); each block
       typically describes one role.
    3. Within each block:
       a. Find a line that contains a date range — this anchors the block as
          a real job entry and yields the duration (years).
       b. Find a line that contains a recognised job-title keyword.
       c. Look for the company on the line immediately adjacent to the title
          (above or below) that does NOT itself look like a title or date line.
    4. Fall back to spaCy ORG extraction only when the adjacent-line approach
       fails, to avoid the false-positive "at Python" pattern of the old code.
    5. Deduplicate and return at most 5 records.
    """

    # ------------------------------------------------------------------
    # Step 1: Isolate the Experience section
    # ------------------------------------------------------------------
    EXP_SECTION_HEADERS = re.compile(
        r'^\s*((?:work\s+)?experiences?|employment(?:\s+history)?|work\s+history'
        r'|professional\s+(?:experience|background)|career\s+history'
        r'|teaching\s+experience|academic\s+experience|faculty\s+experience'
        r'|education\s+experience|positions?\s+held)\s*:?\s*$',
        re.IGNORECASE
    )
    NEXT_SECTION_HEADERS = re.compile(
        r'^\s*(education(?:al)?(?:\s+background)?|skills?|certifications?'
        r'|projects?|awards?|achievements?|interests?|references?|languages?|publications?'
        r'|summary|objective|profile|activities|affiliations?|seminars?(?:\s+and\s+trainings?)?'
        r'|personal\s+(?:information|data)|leadership\s*(?:&|and)\s*activities|volunteer)\s*:?\s*$',
        re.IGNORECASE
    )

    all_lines = text.split('\n')
    exp_lines = []
    in_exp_section = False
    found_exp_section = False
    found_teaching_section = False

    for raw_line in all_lines:
        line = raw_line.strip()
        if EXP_SECTION_HEADERS.match(line):
            in_exp_section = True
            found_exp_section = True
            found_teaching_section = bool(re.search(r'(?i)teaching|faculty|academic', line))
            continue
        if in_exp_section and NEXT_SECTION_HEADERS.match(line):
            break
        if in_exp_section:
            exp_lines.append(raw_line)   # keep blank lines as block separators

    # If no explicit section found, use all lines (graceful fallback)
    scan_lines = exp_lines if exp_lines else all_lines

    # ------------------------------------------------------------------
    # Step 2: Split into blocks on job-entry boundaries
    # ------------------------------------------------------------------
    DATE_RANGE_RE = re.compile(
        r'\b(19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*(Present|19\d{2}|20\d{2})\b',
        re.IGNORECASE
    )
    MONTH_DATE_RANGE_RE = re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\s*(?:-|–|—|to)\s*(?:Present|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{4})\b',
        re.IGNORECASE
    )
    MONTH_YEAR_ONLY_RE = re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b',
        re.IGNORECASE
    )
    MONTH_TOKEN = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    MONTH_DATE_RANGE_RE = re.compile(
        rf'\b{MONTH_TOKEN}\s+\d{{2,4}}\s*(?:-|â€“|â€”|to)\s*(?:Present|Current|Till\s+date|{MONTH_TOKEN}\s+\d{{2,4}}|\d{{2,4}})\b',
        re.IGNORECASE
    )
    MONTH_DATE_RANGE_RE = re.compile(
        rf'\b{MONTH_TOKEN}\s+\d{{2,4}}\s*(?:-|–|—|to)\s*(?:Present|Current|Till\s+date|{MONTH_TOKEN}\s+\d{{2,4}}|\d{{2,4}})\b',
        re.IGNORECASE
    )
    EXPLICIT_YEARS_RE = re.compile(r'\b(\d+(?:\.\d+)?)\+?\s*years?\b', re.IGNORECASE)
    MONTH_DATE_RANGE_RE = re.compile(
        rf'\b{MONTH_TOKEN}\s+\d{{2,4}}\s*(?:-|\\u2013|\\u2014|to)\s*(?:Present|Current|Till\s+date|{MONTH_TOKEN}\s+\d{{2,4}}|\d{{2,4}})\b',
        re.IGNORECASE
    )

    # Final normalized range matcher. It supports year-only, month-name, and
    # numeric month/year formats, including two-digit years and current roles.
    DATE_TOKEN = rf'(?:{MONTH_TOKEN}\.?[\s,./-]*\d{{2,4}}|\d{{1,2}}[/-]\d{{2,4}}|(?:19|20)\d{{2}})'
    DATE_RANGE_RE = re.compile(
        rf'(?P<start>{DATE_TOKEN})\s*(?:-|\u2013|\u2014|to)\s*'
        rf'(?P<end>Present|Current|Till\s+date|To\s+date|{DATE_TOKEN})\b',
        re.IGNORECASE
    )
    MONTH_DATE_RANGE_RE = DATE_RANGE_RE
    EXPLICIT_MONTHS_RE = re.compile(r'\b(\d+(?:\.\d+)?)\+?\s*months?\b', re.IGNORECASE)

    # Job-title keywords (used to detect title lines)
    TITLE_KEYWORDS = re.compile(
        r'\b(developer|engineer|manager|analyst|consultant|architect|lead|'
        r'specialist|programmer|administrator|officer|designer|director|staff|member|teaching|'
        r'validator|statistician|'
        r'coordinator|supervisor|intern|associate|executive|scientist|teacher|writer|'
        r'instructor|professor|lecturer|faculty|principal|dean|tutor|counselor|counsellor|'
        r'registrar|librarian|educator|trainer|adviser|advisor|head\s+teacher|'
        r'guidance\s+counselor|guidance\s+counsellor|curriculum\s+developer|'
        r'academic\s+coordinator|academic\s+adviser|academic\s+advisor|'
        r'school\s+administrator|teaching\s+assistant|research\s+assistant|'
        r'accountant|auditor|bookkeeper|cashier|recruiter|recruitment|human\s+resources|'
        r'nurse|physician|doctor|dentist|pharmacist|therapist|caregiver|medical\s+assistant|'
        r'sales|marketing|representative|agent|customer\s+service|receptionist|secretary|'
        r'clerk|assistant|aide|technician|operator|mechanic|electrician|driver|'
        r'chef|cook|baker|waiter|waitress|server|housekeeper|security|foreman|worker)\b',
        re.IGNORECASE
    )
    TITLE_LINE_RE = re.compile(
        r'^(?P<company>.+?)\s{2,}(?P<title>[A-Za-z].{0,80})$',
        re.IGNORECASE
    )
    TITLE_AT_COMPANY_RE = re.compile(
        r'^(?P<title>[^|]{2,80}?)\s+(?:at|@)\s+(?P<company>[^|]{2,100})$',
        re.IGNORECASE
    )
    CLIENT_LINE_RE = re.compile(r'^\s*(?:Client|Company|Employer|School|Institution|University|College|Organization|Organisation)\s*:\s*(?P<value>.+)$', re.IGNORECASE)
    ROLE_LINE_RE = re.compile(r'^\s*(?:Role|Position|Job\s*Title|Designation|Title)\s*:\s*(?P<title>.+)$', re.IGNORECASE)
    LOCATION_LINE_RE = re.compile(r'\bLocation\s*:\s*(?P<location>.+?)(?=\s+(?:Role|Description|Responsibilities)\s*:|$)', re.IGNORECASE)

    # Lines that are almost certainly NOT a company name or role summary.
    NOT_A_COMPANY_RE = re.compile(
        r'(\d{4}|responsibilities|duties|achieved|managed|developed|led|'
        r'built|created|implemented|worked|collaborated|responsible|'
        r'experience|skills|technologies|frameworks|tools|languages|platforms|'
        r'gmail|yahoo|outlook|hotmail|email|phone|linkedin|github)',
        re.IGNORECASE
    )

    BULLET_PREFIX_RE = re.compile(r'^[\-\*•\u2022]\s*')

    def _is_probable_skill_line(line_text):
        cleaned = BULLET_PREFIX_RE.sub('', line_text).strip()
        if not cleaned:
            return False
        if EMAIL_RE.search(cleaned) or re.search(r'https?://|www\.', cleaned, re.IGNORECASE):
            return True
        if len(cleaned) > 120:
            return False
        if re.search(r'\b(?:experience|skills|technologies|tools|frameworks|languages|platforms)\b', cleaned, re.IGNORECASE):
            return True
        if re.search(r'(?=.*\b(?:java|python|sql|html|css|javascript|react|spring|aws|docker|kubernetes|rest|api|postgres|mysql|oracle|git|jira)\b)(?=.*(?:/|,|\s+and\s+))', cleaned, re.IGNORECASE):
            return True
        if cleaned.count('/') >= 1 and len(cleaned.split()) <= 5:
            return True
        return False

    def _is_probable_title(line_text):
        cleaned = BULLET_PREFIX_RE.sub('', line_text).strip(' -•')
        if not cleaned or len(cleaned) > 140:
            return False
        if EMAIL_RE.search(cleaned) or re.search(r'https?://|www\.', cleaned, re.IGNORECASE):
            return False
        if re.search(r'\b(?:experience|skills|responsibilities|duties|summary|objective|profile|education|certifications|projects|awards)\b', cleaned, re.IGNORECASE):
            return False
        if not TITLE_KEYWORDS.search(cleaned):
            return False
        if re.search(r'[:;]', cleaned):
            return False
        if re.search(r'\b(?:using|with|for|and|or)\b', cleaned) and len(cleaned.split()) > 7 and not TITLE_LINE_RE.match(cleaned):
            return False
        return True

    def _normalize_year(year_text):
        year = int(year_text)
        if year < 100:
            return 2000 + year if year <= 40 else 1900 + year
        return year

    def _date_ordinal(date_text):
        value = date_text.strip().rstrip('.').lower()
        if re.fullmatch(r'(?:present|current|till\s+date|to\s+date)', value):
            now = datetime.now()
            return now.year * 12 + now.month - 1

        numeric = re.fullmatch(r'(\d{1,2})[/-](\d{2,4})', value)
        if numeric:
            month = int(numeric.group(1))
            if not 1 <= month <= 12:
                return None
            year = _normalize_year(numeric.group(2))
            return year * 12 + month - 1

        month_named = re.fullmatch(r'([a-z]+)\.?[\s,./-]*(\d{2,4})', value)
        if month_named:
            month_lookup = {
                'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
                'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
                'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
                'dec': 12, 'december': 12,
            }
            month = month_lookup.get(month_named.group(1))
            if month:
                year = _normalize_year(month_named.group(2))
                return year * 12 + month - 1

        if re.fullmatch(r'(?:19|20)\d{2}', value):
            return int(value) * 12
        return None

    def _years_from_range(line_text):
        m = DATE_RANGE_RE.search(line_text)
        if m:
            start = _date_ordinal(m.group('start'))
            end = _date_ordinal(m.group('end'))
            if start is not None and end is not None and end >= start:
                # Preserve short contracts instead of dropping same-year roles.
                return round(max(end - start, 1) / 12.0, 2)
        m2 = EXPLICIT_YEARS_RE.search(line_text)
        if m2:
            return float(m2.group(1))
        m3 = EXPLICIT_MONTHS_RE.search(line_text)
        if m3:
            return round(float(m3.group(1)) / 12.0, 2)
        return None

    def _is_probable_job_start(line_idx, line_text, prev_text=None, next_text=None,
                               second_next_text=None):
        line = line_text.strip()
        if not line:
            return False
        if CLIENT_LINE_RE.match(line):
            return True
        if ROLE_LINE_RE.match(line):
            return False
        if len(line) > 140:
            return False
        if line.upper().startswith(('RESPONSIBILITIES:', 'ENVIRONMENT:', 'SKILLS:', 'TECHNICAL', 'EDUCATION:', 'SUMMARY:')):
            return False
        if DATE_RANGE_RE.search(line):
            if TITLE_KEYWORDS.search(line):
                return True
            # Common layout: "Company, Location  Date" followed by "Role:".
            return bool(next_text and ROLE_LINE_RE.match(next_text))
        if MONTH_DATE_RANGE_RE.search(line):
            return False
        if TITLE_LINE_RE.match(line):
            prev_ok = prev_text is None or not DATE_RANGE_RE.search(prev_text) and not MONTH_DATE_RANGE_RE.search(prev_text)
            next_ok = next_text is None or DATE_RANGE_RE.search(next_text) or MONTH_DATE_RANGE_RE.search(next_text)
            return prev_ok and next_ok and bool(TITLE_KEYWORDS.search(line))
        if TITLE_KEYWORDS.search(line) and any(
            candidate and DATE_RANGE_RE.search(candidate)
            for candidate in (next_text, second_next_text)
        ):
            return True
        if line.endswith(':'):
            return False
        return False

    def _clean_company(raw):
        """Strip trailing date noise, pipes, hyphens, and whitespace."""
        if EMAIL_RE.search(raw or '') or re.search(r'https?://|www\.', raw or '', re.IGNORECASE):
            return ''
        # Remove a trailing date range or year
        raw = re.sub(r'^\s*(?:Client|Company|Employer|School|Institution|University|College|Organization|Organisation)\s*:\s*', '', raw, flags=re.IGNORECASE)
        raw = LOCATION_LINE_RE.sub('', raw)
        raw = DATE_RANGE_RE.sub('', raw)
        raw = MONTH_DATE_RANGE_RE.sub('', raw)
        raw = re.sub(r'\(\s*\)', '', raw)
        raw = re.sub(r'\b(19|20)\d{2}\b', '', raw)
        raw = re.sub(rf'\b{MONTH_TOKEN}\s*$', '', raw, flags=re.IGNORECASE)
        # Remove common separators that precede a date
        raw = re.sub(r'[\|•·–—\-,;]+\s*$', '', raw)
        raw = re.sub(r'\s+', ' ', raw)
        return raw.strip(' .,;:|')

    def _split_company_location(raw):
        cleaned = _clean_company(raw)
        if not cleaned:
            return None, None

        parts = [part.strip(' .') for part in cleaned.split(',') if part.strip(' .')]
        company = parts[0] if parts else cleaned
        location_parts = parts[1:]

        # Keep legal suffixes attached when they are separated by a comma.
        if location_parts and re.fullmatch(
            r'(?i)(?:inc\.?|llc|ltd\.?|limited|corp\.?|corporation|pvt\.?\s+ltd\.?)',
            location_parts[0],
        ):
            company = f"{company}, {location_parts.pop(0)}"

        # Handle extraction artifacts such as "IBM DALLAS,TEXAS", where the
        # company and city were concatenated without a delimiter.
        doc = nlp(cleaned)
        orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == 'ORG']
        places = [ent.text.strip() for ent in doc.ents if ent.label_ in {'GPE', 'LOC'}]
        if orgs and places and company.lower().startswith(orgs[0].lower()):
            company_remainder = company[len(orgs[0]):].strip(' ,.-')
            company = orgs[0]
            if company_remainder:
                location_parts.insert(0, company_remainder)

        location = ', '.join(location_parts).strip(' ,.-') or None
        if not location and places:
            unique_places = list(dict.fromkeys(places))
            location = ', '.join(unique_places)
        return company[:255], location[:120] if location else None

    def _extract_location(line_text):
        m = LOCATION_LINE_RE.search(line_text)
        if not m:
            return None
        return m.group('location').strip(' .,;:|')[:120]

    def _extract_company_spacy(line_text):
        doc = nlp(line_text)
        orgs = []
        for ent in doc.ents:
            org = ent.text.strip()
            if ent.label_ != 'ORG':
                continue
            if EMAIL_RE.search(org) or re.search(r'https?://|www\.|\.(com|net|org)\b', org, re.IGNORECASE):
                continue
            if NOT_A_COMPANY_RE.search(org) or TECH_OR_ROLE_RE.search(org):
                continue
            orgs.append(org)
        return orgs[0] if orgs else None

    def _clean_job_title(raw):
        value = DATE_RANGE_RE.sub('', raw or '')
        value = re.sub(r'\(\s*\)', '', value)
        value = re.sub(r'^\s*(?:Role|Position|Job\s*Title|Designation|Title)\s*:\s*', '', value, flags=re.IGNORECASE)
        value = re.sub(r'\s+', ' ', value)
        return value.strip(' .,-|')

    def _split_location_prefix_from_title(title):
        location_prefix = (
            r'Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|'
            r'Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|'
            r'Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|'
            r'Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|'
            r'New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|'
            r'Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|'
            r'South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|'
            r'West\s+Virginia|Wisconsin|Wyoming'
        )
        match = re.match(rf'^({location_prefix})\s+(.+)$', title, flags=re.IGNORECASE)
        if match and TITLE_KEYWORDS.search(match.group(2)):
            return match.group(2).strip(), match.group(1).strip()
        return title, None

    def _strip_location_prefix_from_title(title):
        cleaned_title, _ = _split_location_prefix_from_title(title)
        return cleaned_title

    def _parse_inline_role_company(line_text):
        without_dates = DATE_RANGE_RE.sub('', line_text or '')
        without_dates = re.sub(
            r'\([^)]*\)',
            lambda match: '' if DATE_RANGE_RE.search(match.group(0)) else match.group(0),
            without_dates,
        )
        without_dates = re.sub(r'\b(?:Dates?|Period|Duration)\s*:\s*$', '', without_dates, flags=re.IGNORECASE)
        without_dates = re.sub(r'\s+', ' ', without_dates).strip(' .,-|')
        if not without_dates:
            return None, None, None

        if '|' in without_dates:
            parts = [part.strip(' .,-') for part in without_dates.split('|') if part.strip(' .,-')]
        elif re.search(r'\s+[–—-]\s+', without_dates):
            parts = [part.strip(' .,-') for part in re.split(r'\s+[–—-]\s+', without_dates) if part.strip(' .,-')]
        else:
            parts = [part.strip(' .,-') for part in without_dates.split(',') if part.strip(' .,-')]

        if len(parts) < 2:
            return None, None, None

        title_indexes = [
            index for index, part in enumerate(parts)
            if TITLE_KEYWORDS.search(part) and _is_probable_title(part)
        ]
        if not title_indexes:
            return None, None, None

        title_index = title_indexes[0]
        title, title_location = _split_location_prefix_from_title(
            _clean_job_title(parts[title_index])
        )
        company_candidates = [
            part for index, part in enumerate(parts)
            if index != title_index and not DATE_RANGE_RE.search(part)
        ]
        if not title or not company_candidates:
            return None, None, None

        company, location = _split_company_location(company_candidates[0])
        if len(company_candidates) > 1 and not location:
            location = company_candidates[1][:120]
        if title_location and location:
            title_location_lower = title_location.lower()
            location_lower = location.lower()
            if title_location_lower not in location_lower and location_lower not in title_location_lower:
                location = f"{location}, {title_location}"
        elif title_location:
            location = title_location
        return title, company, location

    # First parse common, deterministic resume layouts around a date anchor.
    # This avoids reversing a title and company when a DOCX table is flattened.
    structured_records = []

    def _plausible_company(value):
        value = (value or '').strip()
        return (
            2 < len(value) <= 120
            and not DATE_RANGE_RE.search(value)
            and not RESUME_SECTION_RE.match(value)
            and not re.match(r'(?i)^(managed|developed|created|implemented|handled|handling|served|prepared|led|responsible|adviser)\b', value)
            and not re.match(r'(?i)^(?:client|company|employer|role|position|job\s*title|product\s*title|designation)\s*:', value)
            and not re.search(r'(?i)\b(?:email|phone|gmail|yahoo|outlook|responsibilities|duties)\b', value)
            and not value.endswith(':')
        )

    def _add_structured(title, company, date_line, location=None):
        title = _clean_job_title(title)
        if not title or re.search(
            r'(?i)dean.?s\s+lister|president.?s\s+lister|with\s+honors|magna\s+cum\s+laude|'
            r'honor\s+society|achievement|award|^(?:handled|handling|served|prepared|adviser)\b', title
        ):
            return
        # Cooperating-teacher lines describe an added responsibility at an
        # already listed employer and should not double-count employment time.
        if re.search(r'(?i)^cooperating\s+teacher$', title):
            return
        years = _years_from_range(date_line)
        if years is None or years <= 0:
            return
        company_parts = re.split(r'\t+|\s{2,}', company, maxsplit=1)
        company_name, inline_location = _split_company_location(company_parts[0])
        if len(company_parts) > 1 and not location:
            location = company_parts[1].strip()
        if not company_name:
            return
        structured_records.append({
            'job_title': title[:80],
            'company': company_name,
            'location': location or inline_location or 'Not Identified',
            'years': years,
        })

    for index, line in enumerate(scan_lines):
        stripped = line.strip()
        date_match = DATE_RANGE_RE.search(stripped)
        if not date_match:
            continue
        remainder = DATE_RANGE_RE.sub('', stripped).strip(' ()[]-–—,;|')
        previous = scan_lines[index - 1].strip() if index > 0 else ''
        previous_two = scan_lines[index - 2].strip() if index > 1 else ''

        if remainder and TITLE_KEYWORDS.search(remainder) and _plausible_company(previous):
            _add_structured(remainder, previous, stripped)
            continue
        if TITLE_KEYWORDS.search(previous) and _plausible_company(previous_two):
            _add_structured(previous, previous_two, stripped)
            continue
        if (_plausible_company(previous) and found_teaching_section
                and re.search(r'(?i)\b(?:school|academy|college|university|center|inc\.?)\b', previous)
                and not re.search(r'(?i)teaching\s+internship', previous)):
            _add_structured('Teacher', previous, stripped)
            continue
        if found_exp_section and not remainder:
            company_hint = None
            for back in range(index - 1, max(-1, index - 20), -1):
                candidate = scan_lines[back].strip()
                if re.search(r'(?i)\b(?:school|academy|college|university|center|inc\.?)\b', candidate) and _plausible_company(candidate):
                    company_hint = candidate
                    break
            prior_title = next(
                (record['job_title'] for record in reversed(structured_records)
                 if re.search(r'(?i)teacher|faculty', record['job_title'])),
                None,
            )
            if company_hint and prior_title:
                _add_structured(prior_title, company_hint, stripped)

    # Role, employer, then a date several lines later (common when duties sit
    # between the employer and the school-year line).
    for index, line in enumerate(scan_lines):
        title = line.strip()
        if (not TITLE_KEYWORDS.search(title) or len(title) > 70
                or DATE_RANGE_RE.search(title) or index + 1 >= len(scan_lines)):
            continue
        company = scan_lines[index + 1].strip()
        if not _plausible_company(company):
            continue
        for lookahead in range(index + 2, min(len(scan_lines), index + 22)):
            candidate_date = scan_lines[lookahead].strip()
            if DATE_RANGE_RE.search(candidate_date):
                _add_structured(title, company, candidate_date)
                break

    # A following employer can omit the repeated role title. Reuse the most
    # recent teaching/faculty title, but only for a clearly named institution.
    for index, line in enumerate(scan_lines):
        if not DATE_RANGE_RE.search(line):
            continue
        company_hint = None
        for back in range(index - 1, max(-1, index - 20), -1):
            candidate = scan_lines[back].strip()
            if re.search(r'(?i)\b(?:school|academy|college|university|center|inc\.?)\b', candidate) and _plausible_company(candidate):
                company_hint = candidate
                break
        if not company_hint:
            continue
        company_key = _clean_company(company_hint).lower()
        if any(record['company'].lower() == company_key for record in structured_records):
            continue
        prior_title = next(
            (record['job_title'] for record in reversed(structured_records)
             if re.search(r'(?i)teacher|faculty', record['job_title'])),
            None,
        )
        if prior_title:
            _add_structured(prior_title, company_hint, line)

    # "Pampanga High School Teaching Internship" + date on the next line.
    for index, line in enumerate(scan_lines[:-1]):
        match = re.match(r'(?i)^(.+?\b(?:school|college|university|academy))\s+(teaching\s+internship)$', line.strip())
        if match and DATE_RANGE_RE.search(scan_lines[index + 1]):
            location = DATE_RANGE_RE.sub('', scan_lines[index + 1]).strip(' ()[]-–—,;|') or None
            _add_structured(match.group(2).title(), match.group(1), scan_lines[index + 1], location)

    blocks = []
    current_block = []
    for i, raw_line in enumerate(scan_lines):
        stripped = raw_line.strip()
        prev_line = scan_lines[i - 1].strip() if i > 0 else ''
        next_line = scan_lines[i + 1].strip() if i + 1 < len(scan_lines) else ''
        second_next_line = scan_lines[i + 2].strip() if i + 2 < len(scan_lines) else ''

        if _is_probable_job_start(
            i, stripped, prev_line, next_line, second_next_line
        ):
            if current_block:
                blocks.append(current_block)
            current_block = [stripped]
            continue

        if stripped:
            current_block.append(stripped)
        else:
            if current_block:
                blocks.append(current_block)
                current_block = []
    if current_block:
        blocks.append(current_block)

    # ------------------------------------------------------------------
    # Step 3: Process each block
    # ------------------------------------------------------------------
    records = []
    for block in blocks:
        parsed_company_hint = None
        parsed_location = None
        for bline in block:
            client_match = CLIENT_LINE_RE.match(bline)
            if client_match and not parsed_company_hint:
                parsed_company_hint, inline_location = _split_company_location(
                    client_match.group('value')
                )
                parsed_location = parsed_location or inline_location
            if not parsed_location:
                parsed_location = _extract_location(bline)

        # a. Find date range in the block → confirms this is a job entry + years
        years = 0.0
        date_line_idx = None
        for i, bline in enumerate(block):
            y = _years_from_range(bline)
            if y is not None:
                years = y
                date_line_idx = i
                break

        # b. Find the line that looks most like a job title
        title_line_idx = None
        job_title = None
        for i, bline in enumerate(block):
            cleaned_bline = BULLET_PREFIX_RE.sub('', bline).strip()
            role_match = ROLE_LINE_RE.match(cleaned_bline)
            if role_match:
                title_line_idx = i
                job_title = role_match.group('title').strip()
                continue
            if _is_probable_title(cleaned_bline):
                # Prefer shorter lines (job titles are concise)
                title_match = TITLE_LINE_RE.match(cleaned_bline)
                at_company_match = TITLE_AT_COMPANY_RE.match(cleaned_bline)
                pipe_parts = [part.strip() for part in cleaned_bline.split('|') if part.strip()]
                pipe_title_index = next(
                    (index for index, part in enumerate(pipe_parts) if TITLE_KEYWORDS.search(part)),
                    None
                )
                if len(pipe_parts) >= 2 and pipe_title_index is not None:
                    candidate_title = pipe_parts[pipe_title_index]
                    company_parts = [
                        part for index, part in enumerate(pipe_parts)
                        if index != pipe_title_index and not DATE_RANGE_RE.search(part)
                    ]
                    candidate_company = company_parts[0] if company_parts else None
                    if len(company_parts) > 1 and not parsed_location:
                        parsed_location = company_parts[1][:120]
                elif at_company_match:
                    candidate_title = at_company_match.group('title').strip()
                    candidate_company = at_company_match.group('company').strip()
                elif title_match:
                    candidate_title, title_location = _split_location_prefix_from_title(
                        title_match.group('title').strip()
                    )
                    candidate_company = title_match.group('company').strip()
                    if title_location and not parsed_location:
                        parsed_location = title_location
                else:
                    inline_title, inline_company, inline_location = _parse_inline_role_company(cleaned_bline)
                    candidate_title = inline_title or cleaned_bline
                    candidate_company = inline_company
                    parsed_location = parsed_location or inline_location

                if candidate_company:
                    candidate_company, inline_location = _split_company_location(
                        candidate_company
                    )
                    if inline_location and parsed_location:
                        inline_lower = inline_location.lower()
                        parsed_lower = parsed_location.lower()
                        if inline_lower not in parsed_lower and parsed_lower not in inline_lower:
                            parsed_location = f"{inline_location}, {parsed_location}"
                    else:
                        parsed_location = parsed_location or inline_location

                should_use_candidate = title_line_idx is None
                if not should_use_candidate and date_line_idx is not None:
                    current_before_date = title_line_idx <= date_line_idx
                    candidate_before_date = i <= date_line_idx
                    current_distance = abs(title_line_idx - date_line_idx)
                    candidate_distance = abs(i - date_line_idx)
                    should_use_candidate = (
                        (candidate_before_date and not current_before_date)
                        or (
                            candidate_before_date == current_before_date
                            and candidate_distance < current_distance
                        )
                        or (
                            candidate_before_date == current_before_date
                            and candidate_distance == current_distance
                            and len(candidate_title) < len(job_title or '')
                        )
                    )
                elif not should_use_candidate:
                    should_use_candidate = len(candidate_title) < len(job_title or '')

                if should_use_candidate:
                    title_line_idx = i
                    job_title = candidate_title
                    parsed_company_hint = parsed_company_hint or candidate_company

        if job_title is None or years <= 0:
            # Require a real date/year cue before accepting a block as an
            # experience entry. Otherwise skill bullets and summaries can be
            # misread as jobs.
            continue

        # Clean up the title: strip the date range portion if on the same line
        job_title_clean = DATE_RANGE_RE.sub('', job_title).strip(' –—-|,;')
        job_title_clean = MONTH_DATE_RANGE_RE.sub('', job_title_clean).strip(' â€“â€”-|,;')
        job_title_clean = _clean_job_title(job_title_clean)
        # Trim to max 80 chars
        job_title_clean = job_title_clean[:80].strip()

        # c. Find the company: check the line immediately above/below the title line
        company = parsed_company_hint
        candidates = []
        if title_line_idx > 0:
            candidates.append(block[title_line_idx - 1])
        if title_line_idx < len(block) - 1:
            candidates.append(block[title_line_idx + 1])

        if company:
            company = _clean_company(company)
            comma_parts = [p.strip() for p in company.split(',') if p.strip()]
            if len(comma_parts) > 1 and len(comma_parts[0]) <= 80:
                company = comma_parts[0]

        if not company:
            for candidate_line in candidates:
                cleaned_candidate = BULLET_PREFIX_RE.sub('', candidate_line).strip()
                # Skip if it's a date line, another title line, or a duty bullet
                if DATE_RANGE_RE.search(cleaned_candidate):
                    continue
                if _is_probable_title(cleaned_candidate) and cleaned_candidate != job_title:
                    continue
                if _is_probable_skill_line(cleaned_candidate):
                    continue
                if NOT_A_COMPANY_RE.search(cleaned_candidate):
                    continue
                # Skip very long lines (likely duty descriptions)
                if len(cleaned_candidate) > 80:
                    continue
                cleaned = _clean_company(cleaned_candidate)
                if cleaned and len(cleaned) > 1:
                    company, inline_location = _split_company_location(cleaned)
                    parsed_location = parsed_location or inline_location
                    break

        # If adjacent-line search failed, try spaCy on the title line itself
        if not company:
            company = _extract_company_spacy(job_title)

        # If still nothing, try the date line (sometimes "Company Name | 2019–2022")
        if not company and date_line_idx is not None:
            date_line = block[date_line_idx]
            # Strip the date range and see if anything meaningful remains
            leftover, inline_location = _split_company_location(date_line)
            if leftover and len(leftover) > 2 and not TITLE_KEYWORDS.search(leftover):
                company = leftover
                parsed_location = parsed_location or inline_location

        records.append({
            'job_title': job_title_clean,
            'company': company or 'Not Identified',
            'location': parsed_location or 'Not Identified',
            'years': years,
        })

    # ------------------------------------------------------------------
    # Step 4: Deduplicate and keep the complete work history
    # ------------------------------------------------------------------
    unique_records = []
    seen = set()
    record_candidates = structured_records if structured_records else records
    for rec in record_candidates:
        if (not rec.get('job_title') or re.search(
            r'(?i)dean.?s\s+lister|president.?s\s+lister|with\s+honors|magna\s+cum\s+laude',
            rec.get('job_title') or ''
        )):
            continue
        key = (rec['job_title'].lower(), rec['company'].lower())
        if key not in seen:
            seen.add(key)
            unique_records.append(rec)
    return unique_records[:25]
