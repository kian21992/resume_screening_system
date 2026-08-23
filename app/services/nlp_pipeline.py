import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from difflib import SequenceMatcher
from datetime import datetime
from app.services.education_domain import (
    classify_section_heading,
    validate_education_record,
    validate_experience_record,
)

# Load models
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

EMAIL_RE = re.compile(r'(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])')
PHONE_RE = re.compile(
    r'(?<!\d)(?:'
    r'\+\d{1,3}(?:[- .]?\d){7,12}'
    r'|(?:\+?63|0)[- .]?9\d{2}(?:[- .]?\d){7}'
    r'|'
    r'(?:\+?63|0)[- .]?(?:9\d{2}|2)[- .]?\d{3}[- .]?\d{4}'
    r'|(?:\+?\d{1,3}[- .]?)?\(?\d{2,4}\)?[- .]?\d{3}[- .]?\d{4}'
    r')(?!\d)'
)

RESUME_SECTION_RE = re.compile(
    r'^\s*(objective|summary|profile|professional\s+summary|technical\s+skills|skills|'
    r'education|scholastic\s+records?|work(?:ing)?\s+experience|professional\s+experience|teaching\s+experience|'
    r'academic\s+experience|faculty\s+experience|experience|projects?|'
    r'certifications?|certificates?|licenses?|licensure|eligibility|trainings?|seminars?|awards?|'
    r'achievements?|references?|interests?|activities|affiliations?|publications?|'
    r'personal\s+(?:information|data)|responsibilities|environment|languages|langues|tools)\s*:?\s*$',
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
    r'curriculum\s+vitae|resume|application|word|microsoft\s+word|document|docx?|pdf|'
    r'university|college|school|institute|'
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
    source_lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines() if line.strip()]
    # Post-nominal credentials inside CHARACTER REFERENCES belong to the
    # referee, not the applicant. Exclude that section from every credential
    # detection path instead of trying to blacklist individual abbreviations.
    lines = []
    current_section = 'header'
    for line in source_lines:
        section = classify_section_heading(line)
        if section:
            current_section = section
        if current_section != 'references':
            lines.append(line)

    # Reconstruct logical credential rows after PDF extraction wraps them.
    # Pipe-delimited lists usually continue without a bullet, while bulleted
    # lists start a new credential at every bullet. An unmatched parenthesis
    # is also a strong continuation signal (issuer/date often wraps).
    logical_lines = []
    current_section = 'header'
    credential_bullet_re = re.compile(r'^[\s\-*\u2022\u25aa\u25cf\uf0b7\uf06c]+')
    for line in lines:
        section = classify_section_heading(line)
        if section:
            current_section = section
            logical_lines.append(line)
            continue
        if current_section == 'certifications' and logical_lines:
            previous = logical_lines[-1]
            previous_is_heading = classify_section_heading(previous) is not None
            starts_bullet = bool(credential_bullet_re.match(line))
            should_continue = (
                not previous_is_heading
                and not starts_bullet
                and (
                    previous.count('(') > previous.count(')')
                    or '|' in previous
                    or re.search(
                        r'(?i)\b(?:in|of|for|and|on|at|to|the)\s*$',
                        previous,
                    )
                )
            )
            if should_continue:
                logical_lines[-1] = f'{previous} {line}'
                continue
        logical_lines.append(line)
    lines = logical_lines
    records = []

    def add(name, credential_type='Certification', date=None, issuer=None):
        name = re.sub(r'^[\s\-*\u2022\u25aa\u25cf\uf0b7\uf06c]+', '', name or '')
        name = re.sub(r'\s+', ' ', name).strip(' .,:;|-')
        name = re.sub(r',\s*\)$', ')', name)
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

    # Common regulated Philippine credentials and their resume abbreviations.
    known = [
        (r'\bRegistered\s+Psychometrician\b|\bRPm\b', 'Registered Psychometrician', 'Professional License'),
        (r'\bLicensed\s+Professional\s+Teacher\b|\bLET\s*Passer\b|\bLPT\b', 'Licensed Professional Teacher', 'Professional License'),
        (r'\bCertified\s+Professional\s+in\s+Human\s+Resources\b|\bCPHR\b', 'Certified Professional in Human Resources (CPHR)', 'Certification'),
        (r'\bRegistered\s+Nurse\b|\bRN\b', 'Registered Nurse', 'Professional License'),
        (r'\bCertified\s+Public\s+Accountant\b|\bCPA\b', 'Certified Public Accountant', 'Professional License'),
        (r'\bRegistered\s+Criminologist\b|\bRCRIM\b', 'Registered Criminologist', 'Professional License'),
        (r'\bCivil\s+Service\s+(?:(?:Professional\s+)?Eligible|Eligibility)\b|\bCSE\s+Passer\b', 'Civil Service Eligibility', 'Eligibility'),
        (r'\bTESOL\b|\bTeaching\s+English\s+to\s+Speakers\s+of\s+Other\s+Languages\b', 'TESOL Certificate', 'Certification'),
        (r'\bTEFL\b|\bTeaching\s+English\s+as\s+a\s+Foreign\s+Language\b', 'TEFL Certificate', 'Certification'),
        (r'\bNational\s+Certificate\s+(?:Level\s+)?(?:I{1,3}|IV)\b|\bNC\s*(?:II|III|IV|I)\b', 'TESDA National Certificate', 'Certification'),
        (r'\bPRC\s+(?:License|Licensed|ID|Licensure)\b', 'PRC License', 'Professional License'),
    ]
    for pattern, canonical, kind in known:
        for index, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                date_match = date_re.search(line)
                if not date_match and index + 1 < len(lines):
                    next_date = date_re.fullmatch(lines[index + 1].strip(' ()'))
                    date_match = next_date
                add(canonical, kind, date_match.group(0) if date_match else None)

    # Seminars/trainings are credentials too. Support year-first columns and
    # uppercase seminar-title lists while excluding venues and facilitators.
    training_heading = None
    preceding_section = 'header'
    full_date_after_title = re.compile(
        r'(?i)^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
        r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|'
        r'Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:,)?\s+(?:19|20)\d{2}$'
    )
    for index, line in enumerate(lines):
        section = classify_section_heading(line)
        wrapped_credential_word = bool(
            section == 'training'
            and preceding_section == 'certifications'
            and index + 1 < len(lines)
            and full_date_after_title.fullmatch(lines[index + 1].strip(' ()'))
        )
        if section == 'training' and not wrapped_credential_word:
            training_heading = index
            break
        if section and not wrapped_credential_word:
            preceding_section = section
    if training_heading is not None:
        training_date_re = re.compile(
            r'(?i)^(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
            r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|'
            r'Nov(?:ember)?|Dec(?:ember)?)\.?\s+'
            r'\d{1,2}(?:st|nd|rd|th)?(?:\s*[-–—]\s*\d{1,2}(?:st|nd|rd|th)?)?'
            r',?\s+(?:19|20)\d{2}|(?:19|20)\d{2})'
            r'(?:\s+inclusive\b.*)?$'
        )
        venue_re = re.compile(
            r'(?i)\b(?:school|university|college|academy|institute|center|centre|'
            r'hotel|auditorium|atrium|division|department|office|city hall)\b'
        )

        def is_training_venue(value):
            return bool(
                venue_re.search(value or '')
                and not re.search(
                    r'(?i)\b(?:training|seminar|workshop|forum|symposium|program)\b',
                    value or '',
                )
            )

        training_consumed = set()
        section_end = len(lines)
        for position in range(training_heading + 1, len(lines)):
            next_section = classify_section_heading(lines[position])
            if next_section and next_section != 'training':
                section_end = position
                break

        # Recover complete, possibly wrapped titles by walking backward from
        # each date. A venue is metadata, not part of the seminar name.
        for date_index in range(training_heading + 1, section_end):
            date_match = training_date_re.fullmatch(lines[date_index].strip(' ()'))
            if not date_match:
                continue
            cursor = date_index - 1
            issuer = None
            if cursor > training_heading and is_training_venue(lines[cursor]):
                issuer = lines[cursor]
                training_consumed.add(cursor)
                cursor -= 1
            title_parts = []
            title_indexes = []
            while cursor > training_heading and len(title_parts) < 4:
                candidate = lines[cursor]
                if (
                    training_date_re.fullmatch(candidate.strip(' ()'))
                    or classify_section_heading(candidate)
                    or is_training_venue(candidate)
                ):
                    break
                title_parts.insert(0, candidate)
                title_indexes.append(cursor)
                cursor -= 1
                if re.match(r'^(?:19|20)\d{2}\s+\D', candidate):
                    break
            if title_parts:
                add(
                    ' '.join(title_parts),
                    'Training',
                    date_match.group(0),
                    issuer,
                )
                training_consumed.update(title_indexes)
                training_consumed.add(date_index)
                if date_index + 1 < section_end and is_training_venue(lines[date_index + 1]):
                    training_consumed.add(date_index + 1)

        index = training_heading + 1
        while index < len(lines):
            line = lines[index]
            if re.fullmatch(
                r'(?i)(?:school\s+)?designations?\s*(?:/|&|and)\s*leadership\s+roles?',
                line,
            ):
                break
            section = classify_section_heading(line)
            if section and section != 'training':
                break
            if index in training_consumed:
                index += 1
                continue
            year_entry = re.match(
                r'^(?P<year>(?:19|20)\d{2})\s+(?:(?:Participant|Paticipant),?\s*)?(?P<title>.+)$',
                line,
                re.IGNORECASE,
            )
            if year_entry:
                title_parts = [year_entry.group('title').strip()]
                lookahead = index + 1
                while lookahead < len(lines):
                    continuation = lines[lookahead]
                    if (re.match(r'^(?:19|20)\d{2}\s+', continuation)
                            or training_date_re.fullmatch(continuation.strip(' ()'))
                            or classify_section_heading(continuation)):
                        break
                    if re.search(r'(?i)\bpage\s+\d+\b', continuation):
                        lookahead += 1
                        continue
                    title_parts.append(continuation)
                    lookahead += 1
                add(' '.join(title_parts), 'Training', year_entry.group('year'))
                index = lookahead
                continue
            letters = ''.join(char for char in line if char.isalpha())
            uppercase_share = (sum(char.isupper() for char in letters) / len(letters)) if letters else 0
            if (uppercase_share >= 0.8 and 2 <= len(line.split()) <= 20
                    and not re.search(r'(?i)\b(?:city|pampanga|principal)\b', line)):
                add(line, 'Training')
            index += 1

    # Extract labelled certificate lists, including PDF-wrapped continuation
    # lines, before walking the broader section. Track consumed lines so skill
    # and award text in a combined section cannot bleed into certifications.
    handled_compound_lines = set()
    compound_heading_index = next((
        i for i, line in enumerate(lines)
        if re.fullmatch(r'(?i)certifications?\s*[,/&]\s*skills?\s*(?:[,/&]\s*awards?)?', line)
    ), None)
    labelled_index = next((
        i for i, line in enumerate(lines)
        if re.match(r'(?i)^[-*\u2022\u25aa]?\s*certifications?\s*:', line)
    ), None)
    if compound_heading_index is not None and labelled_index is not None:
        handled_compound_lines.update(range(compound_heading_index + 1, labelled_index))
        labelled = re.match(
            r'(?i)^[-*\u2022\u25aa]?\s*certifications?\s*:\s*(.*)$',
            lines[labelled_index],
        )
        certificate_text = labelled.group(1).strip()
        end_index = labelled_index + 1
        while end_index < len(lines) and not re.match(
            r'(?i)^[-*\u2022\u25aa]?\s*(?:skills?|awards?|education|experience|projects?)\s*:',
            lines[end_index],
        ) and not RESUME_SECTION_RE.match(lines[end_index]):
            certificate_text += ' ' + lines[end_index].lstrip('-*\u2022 ').strip()
            end_index += 1
        handled_compound_lines.update(range(labelled_index, end_index))
        for item in re.split(r'\s*;\s*|\s+\|\s+', certificate_text):
            add(item)
        if end_index < len(lines) and re.match(r'(?i)^[-*\u2022\u25aa]?\s*awards?\s*:', lines[end_index]):
            handled_compound_lines.update(range(end_index, len(lines)))

    # Preserve explicitly named certifications not covered by the aliases above.
    # Parse dated entries under combined headings such as "LICENSES &
    # CERTIFICATIONS". Designed PDFs commonly wrap a credential title across
    # two lines and put its date on the following line.
    dated_section_lines = set()
    combined_license_heading = re.compile(
        r'(?i)^(?:licenses?|licensure|eligibility)\s*(?:&|and|/)\s*certifications?$'
        r'|^certifications?\s*(?:&|and|/)\s*(?:licenses?|licensure|eligibility)$'
    )
    broad_date_line = re.compile(
        rf'(?i)^(?:{month})\s+\d{{1,2}}(?:\s*-\s*\d{{1,2}})?(?:,)?\s+(?:19|20)\d{{2}}$'
    )
    heading_index = next((
        index for index, line in enumerate(lines)
        if combined_license_heading.fullmatch(line)
    ), None)
    if heading_index is not None:
        pending_title = []
        for index in range(heading_index + 1, len(lines)):
            line = lines[index]
            next_section = classify_section_heading(line)
            # A wrapped credential may legitimately end in the standalone
            # word "Training"; do not mistake that continuation for a new
            # resume section while inside Licenses & Certifications.
            if next_section and next_section not in {'training', 'certifications'}:
                break
            dated_section_lines.add(index)
            if broad_date_line.fullmatch(line):
                title = ' '.join(pending_title)
                title = re.sub(r'^\d+(?=[A-Za-z])', '', title).strip()
                if title:
                    add(title, 'Certification', line)
                pending_title = []
            elif len(line) <= 120:
                pending_title.append(line)

    section_active = False
    for index, line in enumerate(lines):
        if index in handled_compound_lines or index in dated_section_lines:
            continue
        if re.fullmatch(
            r'(?:professional\s+)?(?:certifications?|certificates?|licenses?|licensure|eligibility)'
            r'|(?:licenses?|licensure|eligibility)\s*(?:&|and|/)\s*certifications?'
            r'|certifications?\s*(?:&|and|/)\s*(?:licenses?|licensure|eligibility)'
            r'(?:\s*[,/&]\s*(?:skills?|awards?))*\s*:?',
            line.rstrip(':').strip(),
            re.IGNORECASE,
        ):
            section_active = True
            continue
        if section_active and re.fullmatch(
            r'(?i)(?:school\s+)?designations?\s*(?:/|&|and)\s*leadership\s+roles?',
            line,
        ):
            section_active = False
            continue
        current_section = classify_section_heading(line)
        if section_active and current_section and current_section != 'certifications':
            section_active = False
            continue
        if section_active and RESUME_SECTION_RE.match(line):
            section_active = False
            continue
        labelled_certifications = re.match(
            r'(?i)^[-*\u2022\u25aa]?\s*certifications?\s*:\s*(.+)$', line
        )
        if section_active and labelled_certifications:
            for item in re.split(r'\s*;\s*|\s+\|\s+', labelled_certifications.group(1)):
                add(item)
            continue
        if section_active and re.match(r'(?i)^[-*\u2022\u25aa]?\s*(?:skills?|awards?)\s*:', line):
            continue

        explicit = re.search(
            r'(?i)\b((?:certified|certification\s+(?:in|on)|licensed|registered)\s+'
            r'[A-Za-z][A-Za-z0-9 &/+.#-]{2,80})', line
        )
        section_candidate = section_active and (len(line) <= 120 or '|' in line)
        if section_candidate and (
            date_re.fullmatch(line.strip(' ()'))
            or re.match(r'(?i)^(?:issued|issuer|date|valid|expires?|expiration)\s*:', line)
            or re.match(r'(?i)^(?:professional\s+)?(?:experience|education|skills?|projects?|awards?|references?)\b', line)
        ):
            section_candidate = False
        if explicit or section_candidate:
            # Inside a dedicated section, preserve meaningful prefixes such as
            # "AWS" in "AWS Certified Cloud Practitioner". Outside a section,
            # keep using the explicit credential phrase to avoid header noise.
            candidate_text = line if section_candidate else explicit.group(1)
            # Compact PDF templates often flatten a visual certificate list
            # into one pipe-separated line. Preserve each credential instead
            # of treating the complete row as one arbitrary certificate.
            candidate_items = (
                re.split(r'\s*\|\s*|\s*;\s*', candidate_text)
                if section_candidate else [candidate_text]
            )
            for candidate in candidate_items:
                date_match = date_re.search(candidate)
                candidate = date_re.sub('', candidate).strip(' .,:;|-')
                candidate = re.sub(r'\(\s*(?:,\s*)?\)', '', candidate)
                candidate = re.sub(r',\s*\)$', ')', candidate)
                # Do not absorb a narrative clause that happens to follow a
                # real credential on the same extracted line.
                candidate = re.split(
                    r'(?i)\s+(?:with|who|and\s+has)\s+(?=(?:over\s+|more\s+than\s+)?\d+\s+years?\b)',
                    candidate,
                    maxsplit=1,
                )[0].strip(' .,:;|-')
                candidate = re.split(
                    r'(?i)\s+with\s+experience\b', candidate, maxsplit=1
                )[0].strip(' .,:;|-')
                if len(candidate.split()) > 16 or re.search(
                    r'(?i)\b(?:responsible\s+for|worked\s+on|developed|managed|objective|summary)\b',
                    candidate,
                ):
                    continue
                if re.fullmatch(
                    r'(?i)(certifications?|certificates?|licenses?|licensure|eligibility)',
                    candidate,
                ):
                    continue
                if re.fullmatch(
                    r'(?i)Professional\s+Regulation\s+Commission(?:\s*\(PRC\))?',
                    candidate,
                ):
                    continue
                if re.search(r'(?i)\b(?:award|achiever|runner[ -]?up)\b', candidate):
                    kind = 'Award'
                elif re.search(r'(?i)\b(?:course|workshop|webinar|training)\b', candidate):
                    kind = 'Training'
                else:
                    kind = ('Professional License' if re.search(
                        r'(?i)licensed|registered|board|passer', candidate
                    ) else 'Certification')
                add(candidate, kind, date_match.group(0) if date_match else None)

    return records

def extract_contact_info(text):
    """
    Extracts name, email, and phone number from resume text. Candidate names
    are ranked from header evidence instead of accepting the first short line.
    """
    name = None
    email = None
    phone = None
    anonymous_candidate_marker = bool(re.search(
        r'(?i)\b(?:teacher\s+)?candidate\s*(?:no\.?|number|#)?\s*\d+\b',
        text,
    ))

    # Contact details under CHARACTER REFERENCES belong to referees rather
    # than the applicant. Keep those lines available to other extractors but
    # exclude them from contact detection.
    identity_lines = []
    current_identity_section = 'header'
    for raw_line in text.splitlines():
        section = classify_section_heading(raw_line.strip())
        if section:
            current_identity_section = section
        if current_identity_section != 'references':
            identity_lines.append(raw_line)
    identity_text = '\n'.join(identity_lines)

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

        # Some PDFs expose each printed glyph twice (for example,
        # "BBOORRHHAANN GGHHEENNNNAAII"). Collapse the duplicated pairs only
        # when every letter in a word follows that pattern, so legitimate
        # doubled letters in ordinary names are left intact.
        def _collapse_doubled_pdf_word(word):
            letters = ''.join(char for char in word if char.isalpha())
            if len(letters) < 6 or len(letters) % 2:
                return word
            if not all(letters[i].lower() == letters[i + 1].lower()
                       for i in range(0, len(letters), 2)):
                return word
            collapsed = []
            pending_letter = False
            for char in word:
                if char.isalpha():
                    if not pending_letter:
                        collapsed.append(char)
                    pending_letter = not pending_letter
                else:
                    collapsed.append(char)
            return ''.join(collapsed)

        value = ' '.join(_collapse_doubled_pdf_word(word) for word in value.split())

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
                # A single-letter name token is an initial even when PDF/DOCX
                # extraction drops its printed period.
                formatted.append(word[0].upper() + '.')
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
        meaningful_words = [
            word.strip(".'-") for word in words
            if len(word.strip(".'-")) > 1
            and word.lower().strip(".'-") not in NAME_PARTICLES | NAME_SUFFIXES
        ]
        if not meaningful_words:
            return False
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
    email_match = EMAIL_RE.search(identity_text)
    if email_match:
        email = email_match.group(0).strip(' .,;:').lower()
        
    # 2. Extract Phone
    phone_match = PHONE_RE.search(identity_text)
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

    # Some highly designed resumes repeat the candidate's proper name beside
    # an applicant-signature label near the end, while the visual header is
    # exported as individually spaced glyphs. That signature is much stronger
    # identity evidence than nearby education/contact labels.
    if not name:
        signature_match = re.search(
            r"(?im)^\s*([^\n]{3,80}?)\s*\n\s*"
            r"(?:APPLICANT(?:'S)?\s+SIGNATURE|SIGNATURE|APPLICANT)\s*$",
            text,
        )
        if signature_match and _is_probable_person_name(signature_match.group(1)):
            name = _format_person_name(_clean_name_candidate(signature_match.group(1)))

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    header_lines = lines[:12]

    # A mononymous candidate can be a valid name. Accept a one-word first line
    # only when the following line is clearly a professional title; this keeps
    # ordinary one-word headings and technologies out of the identity field.
    if (
        not name
        and len(header_lines) >= 2
        and _is_probable_person_name(header_lines[0], allow_single=True)
        and TECH_OR_ROLE_RE.search(header_lines[1])
    ):
        name = _format_person_name(_clean_name_candidate(header_lines[0]))

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

    if not name and not anonymous_candidate_marker:
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

        # A two-column extractor may retain the given/middle names while the
        # surname crosses the gutter. Recover only the missing email suffix
        # when the visible name is an ordered prefix of a concatenated email.
        if email_compact:
            for index, line in enumerate(lines[:80]):
                cleaned = _clean_name_candidate(line)
                if not _is_probable_person_name(cleaned) or len(cleaned.split()) < 2:
                    continue
                visible_letters = ''.join(
                    word.lower().strip(".'-") for word in cleaned.split()
                    if len(word.strip(".'-")) > 1
                )
                if len(visible_letters) >= 3 and email_compact.startswith(visible_letters):
                    missing = email_compact[len(visible_letters):]
                    if len(missing) >= 3 and missing.isalpha():
                        ranked_candidates.append((155 - min(index, 20), f'{cleaned} {missing.title()}'))
                    elif not missing:
                        ranked_candidates.append((170 - min(index, 20), cleaned))

        # Multi-column PDF extraction can put the given name and surname on
        # separate header lines. Join adjacent one-word lines when their
        # combined letters match the email identity (in either name order).
        if email_compact:
            for index in range(min(len(header_lines) - 1, 7)):
                first = _clean_name_candidate(header_lines[index])
                second = _clean_name_candidate(header_lines[index + 1])
                if len(first.split()) != 1 or len(second.split()) != 1:
                    continue
                combined = f'{first} {second}'
                if not _is_probable_person_name(combined):
                    continue
                forward = ''.join(char.lower() for char in combined if char.isalpha())
                reverse = ''.join(char.lower() for char in f'{second}{first}' if char.isalpha())
                identity_similarity = max(
                    SequenceMatcher(None, email_compact, forward).ratio(),
                    SequenceMatcher(None, email_compact, reverse).ratio(),
                )
                if identity_similarity >= 0.88:
                    ranked_candidates.append((160 - index * 5, combined))

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
                identity_words = [
                    word.lower().strip(".'-") for word in cleaned_words
                    if len(word.strip(".'-")) > 1
                ]
                email_substring_matches = sum(
                    1 for word in identity_words if word in email_compact
                )
                if email_compact and email_substring_matches >= 2:
                    score += 35
                # A location or another contact fragment extracted from a
                # URL/email line is weaker identity evidence than a standalone
                # header name.
                if (
                    EMAIL_RE.search(line)
                    or PHONE_RE.search(line)
                    or re.search(r'https?://|www\.', line, re.IGNORECASE)
                ) and email_substring_matches < 2:
                    score -= 30
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
        r'^\s*(?:education(?:al)?(?:\s+(?:background|history|qualification|details))?'
        r'|education(?:al)?\s+attainment'
        r'|academic(?:\s+(?:background|history|qualification))?'
        r'|professional\s+qualifications?'
        r'|qualifications?'
        r'|degrees?)\s*(?::\s*(?P<content>.*))?$',
        re.IGNORECASE
    )
    # Section headers that signal the end of the education section
    NEXT_SECTION_HEADERS = re.compile(
        r'^\s*(experience|professional\s+experience|employment|work\s*history|skills?'
        r'|technical\s+skills|certifications?(?:\s*[,/&]\s*(?:skills?|awards?))*|professional\s+(?:training|certifications?)'
        r'|professional\s+training\s+and\s+certifications?'
        r'|projects?|awards?|achievements?|interests?|references?|languages?|publications?'
        r'|summary|objective|profile|activities|affiliations?|seminars?(?:\s+and\s+trainings?)?'
        r'|personal\s+(?:information|data)|(?:(?:teaching|professional|work)\s+)?experiences?)\s*:?\s*$',
        re.IGNORECASE
    )

    lines = [line.strip() for line in text.split('\n')]
    edu_lines = []
    in_edu_section = False
    found_edu_section = False
    education_level_heading = re.compile(
        r'(?i)^\s*(?:tertiary|secondary|primary|elementary)'
        r'(?:\s+(?:level|education))?\s*:?\s*$'
    )

    for line in lines:
        if not line:
            if in_edu_section:
                edu_lines.append('')   # preserve blank line as context
            continue
        header_match = EDU_SECTION_HEADERS.match(line)
        domain_section = classify_section_heading(line)
        if not header_match and re.match(r'^EDUCATION\b', line):
            # Two-column PDFs may append text from the other column to a
            # section heading (for example, "EDUCATION students, foster...").
            in_edu_section = True
            found_edu_section = True
            continue
        if header_match or domain_section == 'education':
            if found_edu_section and edu_lines and edu_lines[-1] != '':
                edu_lines.append('')
            in_edu_section = True
            found_edu_section = True
            inline_content = (header_match.group('content') or '').strip() if header_match else ''
            if inline_content:
                edu_lines.append(inline_content)
            continue
        # Education histories sometimes place awards between tertiary and
        # secondary schooling. Reopen only on an exact, labelled school level
        # after an Education heading has already established the context.
        if found_edu_section and education_level_heading.fullmatch(line):
            if edu_lines and edu_lines[-1] != '':
                edu_lines.append('')
            in_edu_section = True
            edu_lines.append(line)
            continue
        if in_edu_section and (
            NEXT_SECTION_HEADERS.match(line)
            or (domain_section and domain_section != 'education')
        ):
            in_edu_section = False
            continue
        if in_edu_section:
            edu_lines.append(line)

    # An explicitly present but empty Education section must remain empty.
    # Falling back to the whole resume in that case can turn degree words in a
    # summary or job description into fabricated education rows.
    scan_lines = edu_lines if found_edu_section else [l.strip() for l in lines if l.strip()]

    # Reconstruct wrapped degree names before entity extraction. Narrow sidebar
    # layouts often split "Bachelor of Secondary Education Major in Biological
    # Science" across three visual lines.
    rebuilt_education_lines = []
    index = 0
    while index < len(scan_lines):
        line = scan_lines[index]
        if (re.fullmatch(r'(?i)Bachelor\s+of\s+(?:Secondary|Elementary)', line)
                and index + 1 < len(scan_lines)
                and re.match(r'(?i)^Education\b', scan_lines[index + 1])):
            line = f'{line} {scan_lines[index + 1].strip()}'
            index += 2
            if (index < len(scan_lines)
                    and re.fullmatch(r'(?i)(?:Science|Arts|Mathematics|English|Filipino)', scan_lines[index].strip())):
                line = f'{line} {scan_lines[index].strip()}'
                index += 1
            rebuilt_education_lines.append(line)
            continue
        rebuilt_education_lines.append(line)
        index += 1
    scan_lines = rebuilt_education_lines

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
            r'\b(bachelor(?:\'s|s)?(?:\s+of|\s+in|\s+degree)?'
            r'|college\s+graduate'
            r'|b\.s\.?|b\.a\.?|b\.e\.?|b\.tech\.?|btech'
            r'|bsed|beed|bsie|bsn|bsit|bscs|bsa|bba|bshm|bsba|ab'
            r'|bachelor\s+of\s+(?:science|arts|engineering|technology|commerce|business)'
            r'|bs\s+in\b|ba\s+in\b|be\s+in\b)\b',
            re.I),
         "Bachelor's"),

        (re.compile(r'\bgraduate\s+school\b', re.I), "Graduate Studies"),

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

        # Elementary / primary education is useful in school-sector resumes
        # and is often listed as "Intermediate Education" in Philippine CVs.
        (re.compile(
            r'\b(elementary\s+graduate|elementary\s+school|primary\s+school|primary\s+education|'
            r'intermediate\s+education|elementary\s+education)\b',
            re.I),
         "Elementary Education"),
    ]

    # ------------------------------------------------------------------
    # Step 3: Institution extraction helpers
    # ------------------------------------------------------------------
    INSTITUTION_KEYWORD = (
        r'(?:Universit(?:y|ies)|Colleges?|Colegio|Institutes?|Schools?|Academy|Polytechnic)'
    )
    INSTITUTION_RE = re.compile(
        rf'(?P<name>'
        rf'(?:(?:[A-Za-z0-9][A-Za-z0-9&.\'-]*|of|the)\s+){{0,8}}'
        rf'{INSTITUTION_KEYWORD}'
        rf'(?:\s+(?:of|the|[A-Za-z][A-Za-z&.\'-]*)){{0,6}}'
        rf')',
        re.IGNORECASE,
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

    ADDRESS_TAIL_RE = re.compile(
        r'(?i)\s*(?:[,;|]|\s+-\s+)\s*'
        r'(?:barangay|brgy\.?|city|municipality|province|district|campus|'
        r'[A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+)?,\s*[A-Z][A-Za-z.-]+).*$'
    )

    def _clean_institution_name(value, labelled=False):
        value = re.sub(r'^[\s\-*•]+', '', value or '')
        value = re.sub(
            r'(?i)^\s*(?:school|institution|university|college|colegio|academy)\s*:\s*',
            '',
            value,
        )
        value = re.sub(r'^\s*(?:(?:19|20)\d{2}\s*(?:-|–|—|to)\s*(?:present|(?:19|20)\d{2})|(?:19|20)\d{2})\s*', '', value, flags=re.I)
        value = re.sub(r'\s+', ' ', value).strip(' .,;:|-')
        # Flattened PDF rows may put a credential or strand after the school:
        # ``Polytechnic University ... - Bachelor of ...``. Keep that
        # right-hand education detail out of the institution value.
        value = re.split(
            r'\s*-\s*(?=(?:Bachelor|Master|Doctor|Associate|Diploma|'
            r'Information\s+and\s+Communication\s+Technology\s+Strand|'
            r'STEM|HUMSS|ABM|GAS|TVL)\b)',
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(' .,;:|-')
        # When the regex also captures a preceding degree phrase, retain only
        # the institution introduced by a conventional "from" or "at" cue.
        introduced = re.search(
            rf'(?i)\b(?:from|at)\s+(?P<school>.+?\b{INSTITUTION_KEYWORD}\b(?:\s+(?:of|the|[A-Za-z][A-Za-z&.\'-]*)){{0,6}})',
            value,
        )
        if introduced:
            value = introduced.group('school').strip(' .,;:|-')
        if not labelled:
            # A one-row education layout may flatten the school on the left
            # and its location on the right, for example ``Holy Angel
            # University Angeles City, Philippines``. Only remove a trailing
            # place expression after the institution keyword; this preserves
            # legitimate names such as ``University of the Philippines``.
            institution_word = re.search(rf'\b{INSTITUTION_KEYWORD}\b', value, re.I)
            if institution_word:
                trailing_text = value[institution_word.end():]
                if re.fullmatch(
                    r'\s+(?:[A-Z][A-Za-z.-]*\s+){0,2}[A-Z][A-Za-z.-]*\s+'
                    r'(?:City|Province|State|Region)(?:,\s*[A-Z][A-Za-z .-]+)?',
                    trailing_text,
                ):
                    value = value[:institution_word.end()]
            value = ADDRESS_TAIL_RE.sub('', value).strip(' .,;:|-')
        if not value or len(value) > 120 or JUNK_INSTITUTION.match(value):
            return None
        if INSTITUTION_NOISE_RE.search(value):
            return None
        if re.search(r'(?i)\b(?:responsibilities|duties|dean.?s\s+lister|honors?|awards?)\b', value):
            return None
        # Unlabelled candidates must contain an educational institution word.
        # Labels may safely contain well-known acronyms such as PUP or UP.
        if not labelled and not re.search(rf'\b{INSTITUTION_KEYWORD}\b', value, re.I):
            return None
        return value[:100]

    def _extract_institution(line_text, allow_org_fallback=True):
        """Try to pull a school name from a single line."""
        labelled = re.match(
            r'^\s*(?:school|institution|university|college|colegio|academy)\s*:\s*(.+)$',
            line_text,
            re.IGNORECASE
        )
        if labelled:
            name = re.sub(r'\b(?:19|20)\d{2}\b.*$', '', labelled.group(1)).strip(' .,;:|-')
            name = _clean_institution_name(name, labelled=True)
            if name and len(name) > 2:
                return name
        # Template rows often separate a degree and an acronym-only school by
        # tabs or wide spacing: "Bachelor ...    JNTU, Hyderabad". The visual
        # column boundary provides enough evidence to accept the acronym safely.
        standalone_acronym = re.fullmatch(r'[A-Z][A-Z0-9.&-]{2,11}', line_text.strip())
        if standalone_acronym and standalone_acronym.group(0).casefold() not in {
            'bsed', 'beed', 'bsie', 'bsn', 'bsit', 'bscs', 'bsa', 'bba', 'ab',
            'primary', 'secondary', 'tertiary', 'elementary',
        }:
            return standalone_acronym.group(0)
        columns = [part.strip() for part in re.split(r'\t+|\s{2,}', line_text) if part.strip()]
        if len(columns) >= 2 and not re.search(rf'\b{INSTITUTION_KEYWORD}\b', line_text, re.I):
            for column in columns[1:]:
                acronym = re.match(r'^([A-Z][A-Z0-9.&-]{1,11})(?=\s*(?:,|$))', column)
                if acronym:
                    return acronym.group(1)
        colegio = re.search(r'(?i)\b(Colegio\s+de\s+[A-Za-z .\'-]+)', line_text)
        if colegio:
            name = re.split(r'\t|\s{2,}', colegio.group(1))[0].strip(' .,;:|-')
            return _clean_institution_name(name)
        # Regex: look for University/College/Institute/etc.
        m = INSTITUTION_RE.search(line_text)
        if m:
            name = _clean_institution_name(m.group('name'))
            if name and len(name) > 4:
                return name
        if not allow_org_fallback or INSTITUTION_NOISE_RE.search(line_text):
            return None
        # spaCy ORG fallback
        doc = nlp(line_text)
        orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == 'ORG']
        for org in orgs:
            cleaned_org = _clean_institution_name(org)
            if cleaned_org and len(cleaned_org.split()) <= 10:
                return cleaned_org
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
        description = re.split(
            r'\t+|\s{2,}(?=[A-Z][A-Z0-9.&-]{1,11}(?:\s*,|$))',
            description,
            maxsplit=1,
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
        # Compact templates sometimes append a city to the degree line. Strip
        # only a trailing named place; locations inside a credential remain.
        if len(description.split()) >= 5:
            place_entities = [
                ent for ent in nlp(description).ents
                if ent.label_ in {'GPE', 'LOC'} and ent.end_char == len(description)
            ]
            if place_entities:
                place_start = place_entities[-1].start_char
                prefix = description[:place_start]
                prefix = re.sub(r'(?i)\b(?:metro|city\s+of)\s*$', '', prefix)
                description = prefix.strip(' ,;-')
        return (description or fallback_label)[:255]

    # Application templates sometimes leave their bracketed prompts in the
    # submitted file. Pair each labelled institution with the immediately
    # following credential, but never store the prompt itself as evidence.
    template_records = []
    template_line_indexes = set()
    template_school_re = re.compile(
        r'(?i)^\s*\[(?:Name\s+of\s+College/University|Name\s+of\s+School)\]\s*(?P<value>.*)$'
    )
    for template_index, template_line in enumerate(scan_lines):
        template_match = template_school_re.match(template_line)
        if not template_match:
            continue
        degree_index = template_index + 1
        while degree_index < len(scan_lines) and not scan_lines[degree_index].strip():
            degree_index += 1
        if degree_index >= len(scan_lines):
            continue
        degree_value = scan_lines[degree_index].strip()
        if not any(pattern.search(degree_value) for pattern, _ in DEGREE_PATTERNS):
            continue

        degree_value = re.sub(
            r'(?i)^Bachelor\s+of\s+\[Degree\]\s*(?=Bachelor\b)',
            '',
            degree_value,
        )
        degree_value = re.sub(r'(?i)\[Degree\]\s*', '', degree_value)
        degree_value = re.sub(r'(?i)\bSCINCE\b', 'SCIENCE', degree_value)
        degree_value = re.sub(
            r'\s*\(\s*(?:19|20)\d{2}\s*(?:-|\u2013|\u2014|to)\s*'
            r'(?:19|20)\d{2}\s*\)\s*$',
            '',
            degree_value,
        ).strip(' .,;:|-')

        institution_value = re.sub(
            r'\s*\(\s*(?:19|20)\d{2}\s*(?:-|\u2013|\u2014|to)\s*'
            r'(?:19|20)\d{2}\s*\)\s*$',
            '',
            template_match.group('value').strip(),
        )
        institution = _clean_institution_name(institution_value, labelled=True)
        template_records.append({
            'degree': degree_value,
            'institution': institution or 'Unknown Institution',
        })
        template_line_indexes.update({template_index, degree_index})

    # A common modern layout lists a level heading, its degree/strand, and its
    # school. Parse this as a coherent block when at least three levels are
    # present, including institution names wrapped across visual lines.
    level_first_re = re.compile(
        r'(?i)^(College|Senior\s+High\s+School|Junior\s+High\s+School|Elementary)$'
    )
    level_first_headings = [
        index for index, value in enumerate(scan_lines)
        if level_first_re.fullmatch(value.strip())
    ]
    level_first_records = []
    if len(level_first_headings) >= 3:
        for heading_position, heading_index in enumerate(level_first_headings):
            block_end = (
                level_first_headings[heading_position + 1]
                if heading_position + 1 < len(level_first_headings)
                else len(scan_lines)
            )
            heading = scan_lines[heading_index].strip().casefold()
            raw_block = [value.strip() for value in scan_lines[heading_index + 1:block_end] if value.strip()]
            joined_block = []
            block_index = 0
            while block_index < len(raw_block):
                value = raw_block[block_index]
                while (
                    value.count('(') > value.count(')')
                    and block_index + 1 < len(raw_block)
                ):
                    block_index += 1
                    value = f'{value} {raw_block[block_index]}'
                joined_block.append(value)
                block_index += 1

            institution = next((
                _clean_institution_name(value, labelled=True)
                for value in joined_block
                if re.search(rf'\b{INSTITUTION_KEYWORD}\b', value, re.IGNORECASE)
                and _clean_institution_name(value, labelled=True)
            ), None)
            degree = None
            if heading == 'college':
                for value in joined_block:
                    degree_match = next((
                        (pattern.search(value), label)
                        for pattern, label in DEGREE_PATTERNS
                        if pattern.search(value)
                    ), None)
                    if degree_match:
                        degree = _extract_degree_description(
                            value, degree_match[0], degree_match[1]
                        )
                        break
            elif heading == 'senior high school':
                strand = next((
                    value for value in joined_block
                    if re.search(r'(?i)\bstrand\b', value)
                ), None)
                degree = (
                    f'Senior High School - {strand}' if strand
                    else 'Senior High School'
                )
            elif heading == 'junior high school':
                degree = 'Junior High School'
            else:
                degree = 'Elementary School'

            if degree and institution:
                level_first_records.append({
                    'degree': degree[:255],
                    'institution': institution,
                })

    # Some application-style resumes keep every education entry on one
    # labelled row: ``College: Example College (2020-2024)``.  Treat the rows
    # as a set only when at least two recognized levels are present.  That
    # avoids interpreting an isolated narrative use of ``College:`` as a
    # credential, preserves every named institution, and does not invent a
    # bachelor's program when the resume only says that college was attended.
    labelled_row_re = re.compile(
        r'(?i)^(?P<label>college|senior\s+high\s+school|junior\s+high\s+school|elementary)'
        r'\s*:\s*(?P<institution>.+?)\s*$'
    )
    labelled_row_matches = []
    for line_index, line_value in enumerate(scan_lines):
        labelled_match = labelled_row_re.match(line_value.strip())
        if labelled_match:
            labelled_row_matches.append((line_index, labelled_match))

    labelled_inline_records = []
    labelled_line_indexes = set()
    if len(labelled_row_matches) >= 2:
        labelled_degree = {
            'college': 'College Education',
            'senior high school': 'Senior High School',
            'junior high school': 'Junior High School',
            'elementary': 'Elementary School',
        }
        for line_index, labelled_match in labelled_row_matches:
            institution_value = re.sub(
                r'\s*\(\s*(?:19|20)\d{2}\s*(?:-|\u2013|\u2014|to|/)\s*'
                r'(?:19|20)\d{2}\s*\)\s*$',
                '',
                labelled_match.group('institution'),
                flags=re.IGNORECASE,
            )
            institution = _clean_institution_name(
                institution_value,
                labelled=True,
            )
            if not institution:
                continue
            label = re.sub(
                r'\s+', ' ', labelled_match.group('label').casefold()
            ).strip()
            labelled_inline_records.append({
                'degree': labelled_degree[label],
                'institution': institution,
            })
            labelled_line_indexes.add(line_index)

    # ------------------------------------------------------------------
    # Step 4: Scan lines for degree matches
    # ------------------------------------------------------------------
    EDUCATION_LEVEL_RE = re.compile(
        r'(?i)^\s*(?:[-*\u2022\u25aa]\s*)?'
        r'(?P<level>secondary\s+level|intermediate\s+education|'
        r'elementary\s+level|primary\s+level|primary\s+education)\s*:',
    )
    records = [*labelled_inline_records, *template_records]
    for i, line in enumerate(scan_lines):
        if not line or i in labelled_line_indexes or i in template_line_indexes:
            continue
        level_match = EDUCATION_LEVEL_RE.match(line)
        if level_match:
            level_name = level_match.group('level').casefold()
            matched_degree = (
                'High School'
                if 'secondary' in level_name
                else 'Elementary Education'
            )
            degree_match = level_match
        else:
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

        # In compact education layouts an institution and its dates precede
        # the credential, e.g. "National University 2021-2023" followed by
        # "Senior High School Baliwag, Bulacan". The institution line contains
        # "School" in some cases but must not become a second credential row.
        next_line = scan_lines[i + 1].strip() if i + 1 < len(scan_lines) else ''
        previous_line = scan_lines[i - 1].strip() if i > 0 else ''
        if (
            re.fullmatch(r'(?i)(?:secondary|primary|elementary)\s+school', previous_line)
            and re.search(rf'\b{INSTITUTION_KEYWORD}\b', line, re.I)
        ):
            continue
        institution_with_dates = bool(
            re.search(r'\b(?:19|20)\d{2}\b', line)
            and re.search(rf'\b{INSTITUTION_KEYWORD}\b', line, re.IGNORECASE)
        )
        if institution_with_dates and re.match(
            r'(?i)^(?:senior|junior)?\s*high\s+school\b', next_line
        ):
            continue

        # Try to find institution on the same line first
        credential_only_high_school = bool(re.match(
            r'(?i)^(?:(?:senior|junior)\s+high\s+school\b|high\s+school\s*$)', line
        ))
        credential_only_school_level = bool(re.fullmatch(
            r'(?i)(?:secondary|primary|elementary)\s+school', line
        ))
        exact_high_school_label = bool(re.fullmatch(r'(?i)high\s+school', line))
        credential_only_abbreviation = bool(re.fullmatch(
            r'(?i)(?:bsed|beed|bsie|bsn|bsit|bscs|bsa|bba|bshm|bsba|ab)', line
        ))
        generic_school_graduate = bool(re.fullmatch(
            r'(?i)\s*(?:elementary|high\s+school)\s+graduate\s*', line
        ))
        generic_college_graduate = bool(re.fullmatch(
            r'(?i)\s*(?:college|tertiary)\s+graduate\s*', line
        ))
        generic_graduate_studies = matched_degree == 'Graduate Studies'
        institution = None if (credential_only_high_school or credential_only_school_level or credential_only_abbreviation or generic_college_graduate or generic_school_graduate or generic_graduate_studies) else _extract_institution(
            line, allow_org_fallback=False
        )
        if exact_high_school_label or credential_only_school_level or credential_only_abbreviation or generic_college_graduate or generic_school_graduate or generic_graduate_studies:
            # The institution normally follows this generic level label. Look
            # forward first so a preceding high-school row cannot be borrowed.
            for nearby in scan_lines[i + 1:i + 3]:
                institution = _extract_institution(nearby, allow_org_fallback=False)
                if institution:
                    break

        # If not found, inspect nearby lines in either direction. Education
        # layouts commonly place the school before or after the degree.
        if not institution:
            # Search only within this education entry. A blank line, another
            # degree, or a section heading is a hard boundary; crossing one can
            # attach the neighboring candidate's/school's text to this degree.
            for direction in (-1, 1):
                for distance in (1, 2):
                    j = i + (direction * distance)
                    if not 0 <= j < len(scan_lines):
                        break
                    nearby = scan_lines[j].strip()
                    if not nearby or NEXT_SECTION_HEADERS.match(nearby):
                        break
                    if re.match(r'(?i)^(?:senior|junior)\s+high\s+school\b', nearby):
                        continue
                    nearby_institution = _extract_institution(nearby, allow_org_fallback=False)
                    if nearby_institution and not re.fullmatch(
                        r'(?i)(?:elementary|high\s+school|college|tertiary)\s+graduate|graduate\s+school',
                        nearby,
                    ):
                        institution = nearby_institution
                        break
                    if any(pattern.search(nearby) for pattern, _ in DEGREE_PATTERNS):
                        dated_school = (
                            re.search(r'\b(?:19|20)\d{2}\b', nearby)
                            and re.search(rf'\b{INSTITUTION_KEYWORD}\b', nearby, re.IGNORECASE)
                        )
                        if dated_school:
                            institution = _extract_institution(nearby)
                        break
                    institution = _extract_institution(nearby)
                    if institution:
                        break
                if institution:
                    break

        degree_description = (
            matched_degree
            if level_match
            else _extract_degree_description(line, degree_match, matched_degree)
        )
        if generic_college_graduate:
            specific_bachelor = re.search(
                r'(?i)\bBachelor\s+of\s+[A-Za-z][A-Za-z &-]{2,80}', text
            )
            degree_description = specific_bachelor.group(0).strip() if specific_bachelor else "Bachelor's degree"
        if matched_degree == 'Graduate Studies':
            completed_requirements = any(
                re.search(r'(?i)completed\s+academic\s+requirements?', nearby)
                for nearby in scan_lines[i:i + 3]
            )
            degree_description = ('Graduate Studies (Completed Academic Requirements)'
                                  if completed_requirements else 'Graduate Studies')
        if re.fullmatch(r'(?i)Bachelor\s+of\s+(?:Secondary|Elementary)', degree_description):
            continuation_parts = []
            for detail in scan_lines[i + 1:i + 3]:
                detail = detail.strip(' -*|')
                if not detail or re.search(r'\b(?:19|20)\d{2}\b', detail):
                    break
                if re.match(r'(?i)^(?:Education|Science|Arts|Engineering|Technology|Major\b)', detail):
                    continuation_parts.append(detail)
                elif _extract_institution(detail, allow_org_fallback=False):
                    break
                else:
                    break
            if continuation_parts:
                degree_description = f"{degree_description} {' '.join(continuation_parts)}"[:255]
        if re.search(r'(?i)\bmaster', degree_description) and any(
            re.search(
                r'(?i)(?:present|ongoing|\b\d+\s+units?\b|'
                r'complete(?:d)?(?:\s+comprehensive)?\s+academic\s+requirements?)',
                detail,
            )
            for detail in scan_lines[max(0, i - 1):i + 4]
        ) and 'ongoing' not in degree_description.casefold():
            degree_description = f'{degree_description} (Ongoing/Incomplete)'[:255]
        if credential_only_high_school:
            level_prefix = re.match(r'(?i)^(?:(?:senior|junior)\s+)?high\s+school', line)
            degree_description = level_prefix.group(0).title()

        # Preserve a wrapped specialization without absorbing the next school,
        # date, or resume section into the credential.
        if not level_match:
            for distance in (1, 2):
                detail_index = i + distance
                if detail_index >= len(scan_lines):
                    break
                detail = scan_lines[detail_index].strip()
                if not detail or classify_section_heading(detail):
                    break
                detail_match = re.match(
                    r'(?i)^(?:major|minor|speciali[sz]ation|concentration|field\s+of\s+study)'
                    r'\s*(?:in|:)?\s+.+$',
                    detail,
                )
                if detail_match:
                    normalized_detail = re.sub(r'\s+', ' ', detail).strip(' .,;:|-()')
                    if normalized_detail.casefold() not in degree_description.casefold():
                        degree_description = f'{degree_description}, {normalized_detail}'[:255]
                    break
                if (_extract_institution(detail, allow_org_fallback=False)
                        or re.search(r'\b(?:19|20)\d{2}\b', detail)):
                    break

        degree_description = degree_description.strip(' .,;:|-')
        if degree_description.count('(') > degree_description.count(')'):
            degree_description = degree_description.rstrip(' (')

        records.append({
            'degree': degree_description,
            'institution': institution or 'Unknown Institution'
        })

    if len(level_first_records) == len(level_first_headings) and level_first_records:
        records = level_first_records

    # ------------------------------------------------------------------
    # Step 5: Deduplicate — prefer records that have a real institution
    # ------------------------------------------------------------------
    rebuilt_bachelor = next((
        line for line in scan_lines
        if re.match(r'(?i)^Bachelor\s+of\s+(?:Secondary|Elementary)\s+Education\b', line)
    ), None)
    if rebuilt_bachelor:
        for record in records:
            if re.match(r'(?i)^Bachelor\s+of\s+(?:Secondary|Elementary)', record.get('degree') or ''):
                record['degree'] = re.sub(r'\s+', ' ', rebuilt_bachelor).strip()[:255]
                break

    # Level-labelled school histories may omit a formal credential name and
    # list more than one institution under Primary, Secondary, or Tertiary.
    # Preserve those explicit labels instead of borrowing a degree from some
    # other line or silently dropping the institution.
    level_heading_re = re.compile(
        r'(?i)^(?:primary|elementary|secondary|tertiary)(?:\s+(?:level|education))?\s*:?$'
    )
    level_headings = [
        index for index, value in enumerate(scan_lines)
        if level_heading_re.fullmatch(value.strip())
    ]
    if len(level_headings) >= 2:
        labelled_records = []
        level_degree = {
            'primary': 'Elementary School',
            'elementary': 'Elementary School',
            'secondary': 'High School',
            'tertiary': 'Tertiary Education',
        }
        for heading_position, start in enumerate(level_headings):
            end = (
                level_headings[heading_position + 1]
                if heading_position + 1 < len(level_headings)
                else len(scan_lines)
            )
            normalized_level = re.match(
                r'(?i)^(primary|elementary|secondary|tertiary)',
                scan_lines[start].strip(),
            ).group(1).casefold()
            block_values = scan_lines[start + 1:end]
            if any(
                pattern.search(value)
                for value in block_values
                for pattern, label in DEGREE_PATTERNS
                if label not in {'High School Diploma', 'Elementary Education'}
            ):
                continue
            for value in block_values:
                institution = _extract_institution(
                    value.strip(),
                    allow_org_fallback=False,
                )
                if institution:
                    labelled_records.append({
                        'degree': level_degree[normalized_level],
                        'institution': institution,
                    })
        if labelled_records:
            # Generic level rows supplement specific credentials. Replacing
            # the list here previously discarded master's and bachelor's rows.
            records.extend(labelled_records)

    # Some CV templates explicitly label four education levels. Treat the
    # Junior/Senior High lines as details of "High School Graduate", preserve
    # "College Graduate" literally, and do not infer a degree from work text.
    explicit_labels = {
        'Elementary Graduate', 'High School Graduate',
        'College Graduate', 'Graduate School',
    }
    if {'College Graduate', 'Graduate School'}.issubset(set(scan_lines)):
        literal_records = []
        for index, value in enumerate(scan_lines):
            if value not in explicit_labels or index + 1 >= len(scan_lines):
                continue
            institution_line = scan_lines[index + 1].strip()
            institution = re.sub(
                r'\s*\((?:19|20)\d{2}[^)]*\)\s*$', '', institution_line
            ).strip()
            degree = value
            requirement = re.search(
                r'(?i)\((Completed\s+Academic\s+Requirements?)\)',
                institution_line,
            )
            if value == 'Graduate School' and requirement:
                degree = f'Graduate School ({requirement.group(1)})'
                institution = institution_line[:requirement.start()].strip()
            if institution:
                literal_records.append({'degree': degree, 'institution': institution})
        if len(literal_records) >= 3:
            records = literal_records

    unique_records = []
    seen = set()
    for rec in records:
        is_valid, _ = validate_education_record(rec)
        if not is_valid:
            continue
        normalized_degree = re.sub(r'[^a-z0-9]+', ' ', rec['degree'].casefold()).strip()
        if re.search(r'\b(?:master|m ed)\b', normalized_degree):
            degree_family = 'masters'
        elif re.search(r'\b(?:bachelor|beed|bsed)\b', normalized_degree):
            degree_family = 'bachelors'
        else:
            degree_family = normalized_degree
        semantic_duplicate = next((
            existing for existing in unique_records
            if existing['institution'].casefold() == rec['institution'].casefold()
            and (
                ('masters' if re.search(
                    r'\b(?:master|m[ .]?ed)\b', existing['degree'], re.IGNORECASE
                ) else 'bachelors' if re.search(
                    r'\b(?:bachelor|beed|bsed)\b', existing['degree'], re.IGNORECASE
                ) else re.sub(
                    r'[^a-z0-9]+', ' ', existing['degree'].casefold()
                ).strip())
                == degree_family
            )
        ), None)
        if semantic_duplicate:
            # Prefer the descriptive degree over a later abbreviation-only
            # statement such as "requirements leading to M.Ed".
            if len(rec['degree']) > len(semantic_duplicate['degree']):
                semantic_duplicate.update(rec)
            continue
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
            r'(?ims)^\s*(?:(?:work\s+)?experiences?|employment(?:\s+history)?'
            r'|work\s+history|professional\s+(?:experience|background)'
            r'|career\s+history|teaching\s+experience|academic\s+experience'
            r'|faculty\s+experience|education\s+experience|positions?\s+held)'
            r'\s*:?\s*$\n(?P<body>.*?)(?=^\s*(?:education(?:al)?(?:\s+background)?'
            r'|skills?|certifications?|achievements?|awards?|projects?|interests?'
            r'|languages?|publications?|summary|objective|profile'
            r'|leadership\s*(?:&|and)\s*activities|affiliations?|seminars?'
            r'|references?|personal\s+(?:information|data)|volunteer)\s*:?\s*$|\Z)',
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
        token = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s*\d{4}|(?:19|20)\d{2}'
        range_re = re.compile(rf'(?P<start>{token})\s*(?:-|–|—|to)\s*(?P<end>Present|Current|{token})', re.I)

        def ordinal(value, is_end=False):
            value = value.strip().lower()
            if value in {'present', 'current'}:
                now = datetime.now()
                return now.year * 12 + now.month - 1
            named = re.match(r'([a-z]+)\.?\s*(\d{4})', value)
            if named:
                return int(named.group(2)) * 12 + month_names[named.group(1)[:3]] - 1
            year = int(re.sub(r'\D', '', value))
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

    # 3. Proximity fallback: no explicit statement and no usable dated section
    #    was found. Recover employment periods from date ranges that sit on or
    #    directly next to a job-title line, while EXCLUDING ranges near
    #    education/graduation lines so schooling years are never counted as work.
    #    This only runs as a last resort, so it can add recall but not override
    #    the more precise section-based calculation above.
    MONTH_MAP = {n.lower(): i for i, names in enumerate((
        ('january', 'jan'), ('february', 'feb'), ('march', 'mar'),
        ('april', 'apr'), ('may',), ('june', 'jun'), ('july', 'jul'),
        ('august', 'aug'), ('september', 'sep'), ('october', 'oct'),
        ('november', 'nov'), ('december', 'dec')
    ), start=1) for n in names}
    MTOK = (r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?'
            r'|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?'
            r'|Dec(?:ember)?)\.?')
    DTOK = rf'(?:{MTOK}\s+\d{{4}}|(?:19|20)\d{{2}})'
    PROX_RANGE = re.compile(
        rf'(?P<s>{DTOK})\s*(?:-|\u2013|\u2014|to)\s*(?P<e>Present|Current|{DTOK})',
        re.IGNORECASE)
    TITLE_HINT = re.compile(
        r'(?i)\b(developer|engineer|manager|analyst|consultant|architect|lead'
        r'|specialist|programmer|administrator|officer|designer|director'
        r'|coordinator|supervisor|intern(?:ship)?|associate|executive|scientist|teacher'
        r'|instructor|professor|lecturer|faculty|principal|dean|tutor(?:ing)?|registrar'
        r'|librarian|educator|trainer|adviser|advisor|accountant|auditor|nurse'
        r'|physician|therapist|assistant|technician|staff|clerk|cashier'
        r'|representative|agent|secretary|receptionist|encoder|teller'
        # Parity with TITLE_KEYWORDS so both experience paths agree on what
        # counts as a job line.
        r'|aide|baker|bookkeeper|caregiver|chef|cook|counsellor|counselor'
        r'|dentist|doctor|driver|electrician|foreman|housekeeper|mechanic'
        r'|operator|pharmacist|recruiter|security|server|statistician'
        r'|validator|volunteer|waiter|waitress|worker|writer'
        # Additional common occupations (service, trades, field, creative)
        # that previously scored zero years because the title was unknown.
        r'|barista|bartender|busser|dishwasher|enumerator|vendor|merchandiser'
        r'|salesman|saleslady|welder|painter|plumber|carpenter|janitor'
        r'|utility|messenger|bagger|stocker|warehouseman|laborer|packer'
        r'|tailor|seamstress|barber|stylist|beautician|gardener|landscaper'
        r'|butcher|farmer|fisherman|rider|courier|dispatcher|conductor'
        r'|photographer|videographer|editor|journalist|reporter|translator'
        r'|interpreter|transcriptionist|proofreader|copywriter|illustrator'
        r'|animator|collector|appraiser|underwriter|broker|trader|estimator'
        r'|surveyor|draftsman|inspector|machinist|fabricator|fitter|rigger'
        r'|installer|repairman|serviceman|lineman|liaison)\b')
    EDU_HINT = re.compile(
        r'(?i)\b(bachelor|master|ph\.?d|doctorate|doctor\s+of|associate\s+degree'
        r'|diploma\s+in|graduated|cum\s+laude|magna\s+cum|summa\s+cum|degree\s+in'
        r'|major\s+in|undergraduate|post.?graduate|tertiary\s+education'
        r'|secondary\s+education|elementary\s+education)\b')
    # Award/honor phrases whose date ranges must not be counted as work, even
    # though a token like "dean" (from "Dean's List") looks like a job title.
    AWARD_HINT = re.compile(
        r"(?i)\b(dean'?s\s+list(?:er)?|honor\s+roll|honou?r\s+student"
        r'|with\s+honors?|scholarship|scholar|awardee|recipient|medal(?:ist)?'
        r'|plaque|recognition|distinction|top\s+\d+|rank\s+\d+)\b')

    def _ord_months(value):
        value = value.strip().lower()
        if value in {'present', 'current'}:
            now = datetime.now()
            return now.year * 12 + now.month - 1
        named = re.match(r'([a-z]+)\s+(\d{4})', value)
        if named:
            return int(named.group(2)) * 12 + MONTH_MAP[named.group(1)[:3]] - 1
        return int(value) * 12

    lines = text.split('\n')
    intervals = []
    for i, line in enumerate(lines):
        match = PROX_RANGE.search(line)
        if not match:
            continue
        neighborhood = ' '.join(lines[max(0, i - 1):i + 2])
        # Skip ranges that live near education or award text; require a job
        # title nearby. The award guard prevents honors such as "Dean's List"
        # from being read as employment via the "dean" title keyword.
        if EDU_HINT.search(neighborhood) or AWARD_HINT.search(line):
            continue
        if not TITLE_HINT.search(neighborhood):
            continue
        try:
            start = _ord_months(match.group('s'))
            end = _ord_months(match.group('e'))
        except (ValueError, KeyError):
            continue
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
        r'|work[- ]related\s+experience'
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
        domain_section = classify_section_heading(line)
        if EXP_SECTION_HEADERS.match(line) or domain_section == 'experience':
            if found_exp_section and exp_lines and exp_lines[-1].strip():
                exp_lines.append('')
            in_exp_section = True
            found_exp_section = True
            found_teaching_section = (
                found_teaching_section
                or bool(re.search(r'(?i)teaching|faculty|academic', line))
            )
            continue
        domain_is_heading = bool(
            domain_section
            and (line.isupper() or line.istitle() or line.endswith(':'))
        )
        if in_exp_section and (
            NEXT_SECTION_HEADERS.match(line)
            or (domain_is_heading and domain_section != 'experience')
        ):
            in_exp_section = False
            continue
        if in_exp_section:
            exp_lines.append(raw_line)   # keep blank lines as block separators

    # An explicitly present but empty Experience section must not fall back to
    # scanning education, skills, certificates, or references as employment.
    scan_lines = exp_lines if found_exp_section else all_lines

    # PDF line wrapping can split the final year from a full date range:
    # ``July 31, 2024 - July 31,`` / ``2025``. Rejoin only that narrow pattern
    # so ordinary neighboring lines retain their record boundaries.
    shared_year_month_range_re = re.compile(
        r'(?i)\b(?P<start>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
        r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
        r'Nov(?:ember)?|Dec(?:ember)?)\s*(?:-|\u2013|\u2014|to)\s*'
        r'(?P<end>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
        r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
        r'Nov(?:ember)?|Dec(?:ember)?)\s+(?P<year>(?:19|20)\d{2})\b'
    )
    normalized_scan_lines = []
    line_index = 0
    while line_index < len(scan_lines):
        current_line = scan_lines[line_index]
        current_line = shared_year_month_range_re.sub(
            lambda match: (
                f"{match.group('start')} {match.group('year')} - "
                f"{match.group('end')} {match.group('year')}"
            ),
            current_line,
        )
        next_line = scan_lines[line_index + 1].strip() if line_index + 1 < len(scan_lines) else ''
        if (
            re.search(r'(?i)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
                      r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|'
                      r'Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},\s*$', current_line)
            and re.fullmatch(r'(?:19|20)\d{2}', next_line)
        ):
            normalized_scan_lines.append(f'{current_line} {next_line}')
            line_index += 2
            continue
        normalized_scan_lines.append(current_line)
        line_index += 1
    scan_lines = normalized_scan_lines

    # Preserve a job title wrapped after a trailing comma, such as “Student
    # Internship handled Grade 7, Grade 9,” / “and Grade 10”.
    wrapped_title_lines = []
    line_index = 0
    while line_index < len(scan_lines):
        current_line = scan_lines[line_index]
        next_line = scan_lines[line_index + 1].strip() if line_index + 1 < len(scan_lines) else ''
        if (
            re.search(r'(?i)\binternship\b', current_line)
            and current_line.rstrip().endswith(',')
            and re.match(r'(?i)^and\s+grade\b', next_line)
        ):
            wrapped_title_lines.append(f'{current_line.rstrip()} {next_line}')
            line_index += 2
            continue
        wrapped_title_lines.append(current_line)
        line_index += 1
    scan_lines = wrapped_title_lines

    # ------------------------------------------------------------------
    # Step 2: Split into blocks on job-entry boundaries
    # ------------------------------------------------------------------
    DATE_RANGE_RE = re.compile(
        r'\b(19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*(Present|19\d{2}|20\d{2})\b',
        re.IGNORECASE
    )
    MONTH_DATE_RANGE_RE = re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{4}\s*(?:-|–|—|to)\s*(?:Present|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{4}|\d{4})\b',
        re.IGNORECASE
    )
    MONTH_YEAR_ONLY_RE = re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{4}\b',
        re.IGNORECASE
    )
    MONTH_TOKEN = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?'
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

    # Final normalized range matcher. Full dates must be recognized before
    # month/year forms; otherwise ``June 05, 2019`` can be misread as the year
    # 2005 and inflate a short role into more than a decade of experience.
    FULL_MONTH_DATE = (
        rf'{MONTH_TOKEN}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)(?:19|20)\d{{2}}'
    )
    MONTH_YEAR_DATE = rf'{MONTH_TOKEN}\.?[\s,./-]+\d{{2,4}}'
    DATE_TOKEN = (
        rf'(?:{FULL_MONTH_DATE}|{MONTH_YEAR_DATE}|\d{{1,2}}[/-]\d{{2,4}}|(?:19|20)\d{{2}})'
    )
    DATE_RANGE_RE = re.compile(
            rf'(?P<start>{DATE_TOKEN})\s*(?:-|\u2013|\u2014|to|up\s+to)\s*'
        rf'(?P<end>Present|Current|Till\s+date|To\s+date|{DATE_TOKEN})\b',
        re.IGNORECASE
    )
    MONTH_DATE_RANGE_RE = DATE_RANGE_RE
    SINGLE_YEAR_RE = re.compile(r'\b((?:19|20)\d{2})\b')
    EXPLICIT_MONTHS_RE = re.compile(r'\b(\d+(?:\.\d+)?)\+?\s*months?\b', re.IGNORECASE)

    # Job-title keywords (used to detect title lines)
    TITLE_KEYWORDS = re.compile(
        r'\b(developer|engineer|manager|analyst|consultant|architect|lead|'
        r'specialist|programmer|administrator|officer|designer|director|staff|member|teaching|'
        r'validator|statistician|'
        r'coordinator|supervisor|intern(?:ship)?|volunteer|associate|executive|scientist|teacher|writer|'
        r'instructor|professor|lecturer|faculty|principal|dean|tutor(?:ing)?|counselor|counsellor|'
        r'registrar|librarian|educator|trainer|adviser|advisor|head\s+teacher|'
        r'guidance\s+counselor|guidance\s+counsellor|curriculum\s+developer|'
        r'academic\s+coordinator|academic\s+adviser|academic\s+advisor|'
        r'school\s+administrator|teaching\s+assistant|research\s+assistant|'
        r'accountant|auditor|bookkeeper|cashier|recruiter|recruitment|human\s+resources|'
        r'nurse|physician|doctor|dentist|pharmacist|therapist|caregiver|medical\s+assistant|'
        r'sales|marketing|representative|agent|customer\s+service|receptionist|secretary|'
        r'clerk|assistant|aide|technician|operator|mechanic|electrician|driver|'
        r'chef|cook|baker|waiter|waitress|server|housekeeper|security|foreman|worker|'
        # Additional common occupations so uncommon-but-real job titles are not
        # dropped from experience parsing and ranking.
        r'barista|bartender|busser|dishwasher|enumerator|vendor|merchandiser|'
        r'salesman|saleslady|welder|painter|plumber|carpenter|janitor|utility|'
        r'messenger|bagger|stocker|warehouseman|laborer|packer|tailor|seamstress|'
        r'barber|stylist|beautician|gardener|landscaper|butcher|farmer|fisherman|'
        r'rider|courier|dispatcher|conductor|photographer|videographer|editor|artist|'
        r'correspondent|owner|'
        r'journalist|reporter|translator|interpreter|transcriptionist|proofreader|'
        r'copywriter|illustrator|animator|collector|appraiser|underwriter|broker|'
        r'trader|estimator|surveyor|draftsman|inspector|machinist|fabricator|'
        r'fitter|rigger|installer|repairman|serviceman|lineman|liaison)\b',
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

    BULLET_PREFIX_RE = re.compile(r'^[\-\*•\u2022\u25aa\u2751]\s*')

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

    def _is_duty_detail_line(value):
        """Identify teaching/administrative duty details that mimic roles."""
        return bool(re.match(
            r'(?i)^(?:(?:advisory|subject)\s+teacher\s+for\b|'
            r'teacher\s+adviser\s+for\s+level\b|'
            r'teaching\s+(?:science|english|mathematics|math|filipino|subjects?)\b|'
            r'time\s+spent\b|report\s+directly\b|'
            r'assist(?:ed|ing)?\s+(?:customers?|clients?|the\b)|'
            r'monitor(?:ed|ing)?\s+(?:the\b|office\b))',
            (value or '').strip(),
        ))

    def _is_probable_title(line_text):
        cleaned = BULLET_PREFIX_RE.sub('', line_text).strip(' :-•')
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
        if _is_duty_detail_line(cleaned):
            return False
        detailed_internship_title = bool(re.match(
            r'(?i)^student\s+internship\b.*\bgrade\s+\d+', cleaned
        ))
        if (re.search(r'\b(?:using|with|for|and|or)\b', cleaned)
                and len(cleaned.split()) > 7
                and not TITLE_LINE_RE.match(cleaned)
                and not detailed_internship_title):
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

        full_named = re.fullmatch(
            r'([a-z]+)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*|\s+)((?:19|20)\d{2})',
            value,
        )
        if full_named:
            month_lookup = {
                'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
                'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
                'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
                'dec': 12, 'december': 12,
            }
            month = month_lookup.get(full_named.group(1))
            if month:
                return int(full_named.group(2)) * 12 + month - 1

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
        # A lone year is a common compact-resume convention for short roles.
        # It is only consumed when the surrounding logic has already found a
        # credible role and employer inside the experience section.
        if SINGLE_YEAR_RE.fullmatch(line_text.strip()):
            return 1.0
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
        raw = re.sub(r'^\s*\[\s*(.*?)\s*\]\s*$', r'\1', raw)
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
        org_prefix_has_boundary = bool(
            orgs
            and company.lower().startswith(orgs[0].lower())
            and (
                len(company) == len(orgs[0])
                or company[len(orgs[0])] in ' ,-'
            )
        )
        if orgs and places and org_prefix_has_boundary:
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
        value = BULLET_PREFIX_RE.sub('', raw or '').strip().lstrip(':').strip()
        value = DATE_RANGE_RE.sub('', value)
        value = re.sub(r'\(\s*\)', '', value)
        value = re.sub(
            r'(?i)\s*\(\s*(?:(?:\d+(?:\.\d+)?\s+years?)\s*'
            r'(?:and\s+)?(?:\d+(?:\.\d+)?\s+months?)?|'
            r'\d+(?:\.\d+)?\s+months?)\s*\)\s*$',
            '',
            value,
        )
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
        if re.match(
            r'(?i)^\s*student\s+internship\b.*\bgrade\s+\d+',
            line_text or '',
        ):
            return None, None, None
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

    def _has_organization_hint(value):
        return bool(re.search(
            r'(?i)\b(?:university|college|school|academy|institute|center|centre|'
            r'inc\.?|corporation|company|department|deped|foundation|authority)\b',
            value or '',
        ))

    def _plausible_company(value):
        value = (value or '').strip()
        return (
            2 < len(value) <= 120
            and not BULLET_PREFIX_RE.match(value)
            and not DATE_RANGE_RE.search(value)
            and not RESUME_SECTION_RE.match(value)
            and not re.match(r'(?i)^(?:from|medium\s+of\s+instruction|student(?:\s+age\s+bracket)?|time\s+spent|and|or|to|for|managed|developed|created|implemented|handled|handling|served|prepared|led|built|worked|provided|responsible|adviser|assist(?:ed|ing)?|monitor(?:ed|ing)?|report(?:ed|ing)?|plan(?:ned|ning|s)?|completed|coordinated|encouraged|ensured|observed|supervised|participated|promoted)\b', value)
            and not re.match(r'(?i)^(?:client|company|employer|role|position|job\s*title|product\s*title|project\s*description|designation)\s*:', value)
            and not re.search(r'(?i)\b(?:email|phone|gmail|yahoo|outlook|responsibilities|duties)\b', value)
            and not _looks_like_location_line(value)
            and not value.endswith(':')
        )

    def _looks_like_location_line(value):
        value = (value or '').strip()
        if not value or len(value) > 120:
            return False
        if re.search(
            r'(?i)\b(?:university|college|school|academy|institute|corporation|corp\.?|'
            r'inc\.?|company|group|solutions?|services?)\b',
            value,
        ):
            return False
        if re.fullmatch(
            r'(?i)[A-Za-z .-]{2,60}\s+City\s*,\s*[A-Za-z .-]{2,50}',
            value,
        ):
            return True
        if re.fullmatch(
            r'(?i)[A-Za-z .-]{2,80}\s*,\s*[A-Za-z .-]{2,50}\s*,\s*'
            r'(?:Philippines|United States|UAE|India|Colombia|Canada|Australia)',
            value,
        ):
            return True
        if LOCATION_LINE_RE.search(value):
            return True
        if re.match(
            r'(?i)^(?:purok|barangay|brgy\.?|sitio|district|city\s+of)\b',
            value,
        ):
            return True
        if re.search(
            r'(?i)(?:,\s*(?:Philippines|United States|UAE|India|Colombia|'
            r'[A-Z][A-Za-z .-]+\s+(?:City|Province|State|Region))\s*$)',
            value,
        ):
            return True
        if re.fullmatch(
            r'(?i)(?:Philippines|United States|UAE|India|Colombia|Canada|Australia)',
            value,
        ):
            return True
        return bool(re.fullmatch(
            r'(?i)[A-Za-z .-]{2,50}\s+(?:Pampanga|Bulacan|Manila|Cebu|Davao|'
            r'Laguna|Cavite|Rizal|Tarlac|Bataan|Pangasinan)',
            value,
        ))

    def _add_structured(title, company, date_line, location=None):
        title = _clean_job_title(title)
        if not title or _is_duty_detail_line(title) or re.search(
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

    def _add_undated(title, company='Not Identified', location=None):
        """Preserve explicit experience without inventing a duration."""
        title = _clean_job_title(title)
        company_name, inline_location = _split_company_location(company)
        if not title or not TITLE_KEYWORDS.search(title) or not company_name:
            return
        structured_records.append({
            'job_title': title[:80],
            'company': company_name,
            'location': location or inline_location or 'Not Identified',
            'years': 0.0,
            'duration_unknown': True,
        })

    # Word tables often export an employer row ending in “DURATION”, followed
    # by a role row containing the actual date range. Treat those two rows as
    # one entry so the first employers in long histories are not skipped.
    for index in range(len(scan_lines) - 1):
        employer_row = scan_lines[index].strip()
        role_row = scan_lines[index + 1].strip()
        if not re.search(r'(?i)\bDURATION\s*$', employer_row):
            continue
        if not DATE_RANGE_RE.search(role_row):
            continue
        title = DATE_RANGE_RE.sub('', role_row).strip(' ()[]-–—,;|')
        company = re.sub(r'(?i)\bDURATION\s*$', '', employer_row).strip()
        if (
            _is_probable_title(title)
            and company
            and len(company) <= 120
            and not _is_probable_title(company)
        ):
            company_name, location = _split_company_location(company)
            if company_name:
                _add_structured(title, company_name, role_row, location)

    # Bulleted application-style histories commonly use four consecutive
    # lines for each job: role, employer, location, and date.  The generic
    # block parser can combine repeated entries when there are no blank lines,
    # so capture this strong structure before block parsing.  Strip only the
    # leading list marker; all existing title, employer, location, and duration
    # validation remains in force.
    def _entry_line(value):
        return BULLET_PREFIX_RE.sub('', (value or '').strip()).strip()

    for index in range(len(scan_lines) - 2):
        title = _entry_line(scan_lines[index])
        company = _entry_line(scan_lines[index + 1])
        if (
            not _is_probable_title(title)
            or DATE_RANGE_RE.search(title)
            or not _has_organization_hint(company)
            or not _plausible_company(company)
            or _is_probable_title(company)
        ):
            continue

        location = None
        date_line = None
        third_line = _entry_line(scan_lines[index + 2])
        if DATE_RANGE_RE.search(third_line):
            date_line = third_line
        elif index + 3 < len(scan_lines):
            fourth_line = _entry_line(scan_lines[index + 3])
            if _looks_like_location_line(third_line) and DATE_RANGE_RE.search(fourth_line):
                location = third_line
                date_line = fourth_line

        if date_line:
            _add_structured(title, company, date_line, location)

    # Inline ``Role, Employer, Date`` or ``Employer - Role - Date`` rows.
    for line in scan_lines:
        if not DATE_RANGE_RE.search(line):
            continue
        title, company, location = _parse_inline_role_company(line.strip())
        if title and company:
            _add_structured(title, company, line, location)

    # Date first, followed by ``Role, Employer``. Resolve this before the
    # generic block parser can reverse the neighboring fields.
    for index, date_line in enumerate(scan_lines[:-1]):
        if not DATE_RANGE_RE.search(date_line):
            continue
        date_leftover = DATE_RANGE_RE.sub('', date_line).strip(' ()[]-–—,;|')
        if date_leftover and not re.fullmatch(
            rf'(?i){MONTH_TOKEN}\s+\d{{1,2}}',
            date_leftover,
        ):
            continue
        title, company, location = _parse_inline_role_company(
            scan_lines[index + 1].strip()
        )
        if title and company:
            _add_structured(title, company, date_line, location)

    # An explicit experience section may contain internships or freelance work
    # without dates. Keep those records with an unknown duration so they are
    # visible to reviewers but contribute zero years to scoring.
    if found_exp_section and not any(DATE_RANGE_RE.search(line) for line in scan_lines):
        for index, raw_line in enumerate(scan_lines):
            line = BULLET_PREFIX_RE.sub('', raw_line).strip()
            if not line or DATE_RANGE_RE.search(line) or RESUME_SECTION_RE.match(line):
                continue

            next_line = (
                BULLET_PREFIX_RE.sub('', scan_lines[index + 1]).strip()
                if index + 1 < len(scan_lines) else ''
            )
            if (
                _is_probable_title(line)
                and next_line
                and _plausible_company(next_line)
                and not _is_probable_title(next_line)
                and (
                    _has_organization_hint(next_line)
                    or re.fullmatch(r'\s*\[[^\]]+\]\s*', next_line)
                )
            ):
                _add_undated(line, next_line)
                continue
            organization_like = bool(
                _has_organization_hint(line)
                or re.search(r'(?i)\b(?:DSWD|DepEd|program|department)\b', line)
            )
            if (
                organization_like
                and _plausible_company(line)
                and len(line) <= 80
                and next_line
                and _is_probable_title(next_line)
                and len(next_line.split()) <= 6
                and line.casefold() != next_line.casefold()
            ):
                _add_undated(next_line, line)
                continue

            if re.fullmatch(r'(?i)(?:teaching\s+)?internship', line):
                employer = None
                for detail in scan_lines[index + 1:index + 7]:
                    employer_match = re.search(
                        r'(?i)\bat\s+(.+?\b(?:school|academy|college|university|'
                        r'institute|center|centre))\b',
                        detail,
                    )
                    if employer_match:
                        employer = employer_match.group(1).strip(' .,;:|-')
                        break
                _add_undated(line.title(), employer or 'Not Identified')
                continue

            if (
                re.match(r'(?i)^(?:freelance|self[- ]employed|independent)\b', line)
                and _is_probable_title(line)
            ):
                _add_undated(line, 'Not Identified')

    # Some traditional CVs list several concurrent roles before the date and
    # put the employer after it. Anchor the record to the strong organization
    # line instead of treating neighboring role or location lines as employers.
    for date_index, date_line in enumerate(scan_lines):
        if not DATE_RANGE_RE.search(date_line):
            continue
        company_index = date_index + 1
        while company_index < len(scan_lines) and not scan_lines[company_index].strip():
            company_index += 1
        if company_index >= len(scan_lines):
            continue
        company = scan_lines[company_index].strip()
        if (not _has_organization_hint(company)
                or not _plausible_company(company)
                or _is_probable_title(company)):
            continue

        titles = []
        skipped_non_title = 0
        for position in range(date_index - 1, max(-1, date_index - 6), -1):
            candidate = scan_lines[position].strip()
            if not candidate:
                if titles:
                    break
                continue
            if DATE_RANGE_RE.search(candidate) or RESUME_SECTION_RE.match(candidate):
                break
            if _is_probable_title(candidate):
                titles.append(candidate)
                continue
            if titles:
                break
            skipped_non_title += 1
            if skipped_non_title > 1:
                break
        if not titles:
            continue

        location = None
        if company_index + 1 < len(scan_lines):
            possible_location = scan_lines[company_index + 1].strip()
            if _looks_like_location_line(possible_location):
                location = possible_location
        _add_structured(' / '.join(reversed(titles)), company, date_line, location)

    # Role and date on one line, employer on the next. This is common in clean
    # single-column CVs: ``Backend Developer | Jan 2024 - Apr 2024`` followed
    # by the organization. Remove the date before applying title heuristics so
    # a valid multi-word role is not rejected as narrative text.
    for index, line in enumerate(scan_lines[:-1]):
        stripped = line.strip()
        if not DATE_RANGE_RE.search(stripped):
            continue
        title = DATE_RANGE_RE.sub('', stripped).strip(' []-â€“â€” ,;|')
        company = scan_lines[index + 1].strip()
        if (
            _is_probable_title(title)
            and _plausible_company(company)
            and not _is_probable_title(company)
            and not DATE_RANGE_RE.search(company)
        ):
            _add_structured(title, company, stripped)

    # Role-first layouts: title, date, employer, location. This is common in
    # modern two-column resumes and is more reliable than block proximity.
    for index in range(len(scan_lines) - 1):
        title_line = scan_lines[index].strip()
        date_line = scan_lines[index + 1].strip()
        if (not _is_probable_title(title_line) or DATE_RANGE_RE.search(title_line)
                or not DATE_RANGE_RE.search(date_line)):
            continue
        inline_company = DATE_RANGE_RE.sub('', date_line).strip(' ()-–—,;|')
        if re.fullmatch(r'(?i)(?:blended|remote|online|hybrid|onsite)\s+learning\s+modality', inline_company):
            inline_company = ''
        company = inline_company
        location = None
        if not company and index + 2 < len(scan_lines):
            company = scan_lines[index + 2].strip()
            if index + 3 < len(scan_lines) and _looks_like_location_line(scan_lines[index + 3]):
                location = scan_lines[index + 3].strip()
        if company and _plausible_company(company) and not _is_probable_title(company):
            _add_structured(title_line, company, date_line, location)

    # Date-first layouts sometimes omit the position entirely. Preserve the
    # employment row transparently instead of inventing a role or discarding
    # several years of history.
    for line_index, line in enumerate(scan_lines):
        if not DATE_RANGE_RE.search(line):
            continue
        leftover = DATE_RANGE_RE.sub('', line).strip(' ()-–—,;|')
        adjacent_title = any(
            _is_probable_title(scan_lines[position].strip())
            for position in range(max(0, line_index - 1), min(len(scan_lines), line_index + 2))
            if position != line_index
        )
        nearby_labelled_role = any(
            ROLE_LINE_RE.match(scan_lines[position].strip())
            for position in range(max(0, line_index - 2), min(len(scan_lines), line_index + 3))
            if position != line_index
        )
        nearby_has_title = adjacent_title or nearby_labelled_role
        if (found_exp_section and leftover and not TITLE_KEYWORDS.search(leftover)
                and not nearby_has_title
                and re.search(r'(?i)\b(?:school|academy|college|university|inc\.?|'
                              r'corporation|company|department|deped)\b', leftover)):
            _add_structured('Position Not Stated', leftover, line)

    def _split_role_and_trailing_location(value):
        """Split lines such as 'Student Nurse Manila, Philippines'."""
        cleaned = value.strip()
        title_match = TITLE_KEYWORDS.search(cleaned)
        if not title_match:
            return None, None
        title = cleaned[:title_match.end()].strip(' ,-')
        location = cleaned[title_match.end():].strip(' ,-') or None
        return title, location

    # Compact layout: employer and date on one line, then role and location.
    # Example: "Far Eastern University - Manila 2024-Present" followed by
    # "Student Nurse Manila, Philippines".
    for index, line in enumerate(scan_lines[:-1]):
        stripped = line.strip()
        range_match = DATE_RANGE_RE.search(stripped)
        single_year = re.search(r'\b(?:19|20)\d{2}\s*$', stripped)
        date_match = range_match or single_year
        if not date_match:
            continue
        company = stripped[:date_match.start()].strip(' ,-–—|')
        title, location = _split_role_and_trailing_location(scan_lines[index + 1])
        if title and _plausible_company(company) and not TITLE_KEYWORDS.search(company):
            date_value = date_match.group(0)
            _add_structured(title, company, date_value, location)

    for index, line in enumerate(scan_lines):
        stripped = line.strip()
        date_match = DATE_RANGE_RE.search(stripped)
        if not date_match:
            continue
        remainder = DATE_RANGE_RE.sub('', stripped).strip(' ()[]-–—,;|')
        previous = scan_lines[index - 1].strip() if index > 0 else ''
        previous_two = scan_lines[index - 2].strip() if index > 1 else ''

        at_company_match = TITLE_AT_COMPANY_RE.match(remainder) if remainder else None
        if at_company_match and _is_probable_title(at_company_match.group('title')):
            _add_structured(
                at_company_match.group('title'),
                at_company_match.group('company'),
                stripped,
            )
            continue
        following = scan_lines[index + 1].strip() if index + 1 < len(scan_lines) else ''
        following_is_employer = (
            _plausible_company(following)
            and not _is_probable_title(following)
            and not DATE_RANGE_RE.search(following)
        )
        # Compact rows may put the employer after the date while keeping the
        # role above and the location below: ``2020-2021 - Concentrix``. The
        # surrounding title and location provide enough structure to treat the
        # date-line remainder as the employer without guessing from prose.
        if (
            remainder
            and not remainder.startswith('(')
            and _is_probable_title(previous)
            and _plausible_company(remainder)
            and not _is_probable_title(remainder)
            and following
            and _looks_like_location_line(following)
        ):
            _add_structured(previous, remainder, stripped, following)
            continue
        if (remainder and TITLE_KEYWORDS.search(remainder)
                and _plausible_company(previous) and not following_is_employer):
            _add_structured(remainder, previous, stripped)
            continue
        if (
            TITLE_KEYWORDS.search(previous)
            and not DATE_RANGE_RE.search(previous)
            and _plausible_company(previous_two)
            and not following_is_employer
        ):
            _add_structured(previous, previous_two, stripped)
            continue
        if (_plausible_company(previous) and found_teaching_section
                and re.search(r'(?i)\b(?:school|academy|college|university|center|inc\.?)\b', previous)
                and not any(
                    _is_probable_title(scan_lines[position])
                    for position in range(index - 2, max(-1, index - 5), -1)
                    if not DATE_RANGE_RE.search(scan_lines[position])
                )
                and not re.search(r'(?i)teaching\s+internship', previous)):
            _add_structured('Teacher', previous, stripped)
            continue
        if found_exp_section and not remainder:
            if any(
                _is_probable_title(scan_lines[position])
                for position in range(index - 1, max(-1, index - 4), -1)
                if not DATE_RANGE_RE.search(scan_lines[position])
            ):
                continue
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
        if not _plausible_company(company) or _is_probable_title(company):
            continue
        for lookahead in range(index + 2, min(len(scan_lines), index + 22)):
            candidate_date = scan_lines[lookahead].strip()
            if DATE_RANGE_RE.search(candidate_date):
                location_parts = [
                    scan_lines[position].strip()
                    for position in range(index + 2, lookahead)
                    if _looks_like_location_line(scan_lines[position])
                ]
                _add_structured(
                    title,
                    company,
                    candidate_date,
                    ', '.join(location_parts)[:120] if location_parts else None,
                )
                break

    # Date-anchored layout: company, role, optional location, then date. This
    # complements the role-first pass and prevents the location/date fragment
    # from being selected as the employer.
    for date_index, line in enumerate(scan_lines):
        if not DATE_RANGE_RE.search(line):
            continue
        following_company = (
            scan_lines[date_index + 1].strip()
            if date_index + 1 < len(scan_lines) else ''
        )
        if (
            _has_organization_hint(following_company)
            and _plausible_company(following_company)
            and not _is_probable_title(following_company)
        ):
            continue
        title_index = next((
            position
            for position in range(date_index - 1, max(-1, date_index - 5), -1)
            if (
                not DATE_RANGE_RE.search(scan_lines[position])
                and _is_probable_title(scan_lines[position])
            )
        ), None)
        if title_index is None:
            continue

        company = None
        company_index = None
        for position in (title_index - 1, title_index + 1):
            if not 0 <= position < date_index:
                continue
            candidate = scan_lines[position].strip()
            if _plausible_company(candidate) and not _is_probable_title(candidate):
                company = candidate
                company_index = position
                break
        if not company:
            continue

        location_parts = [
            scan_lines[position].strip()
            for position in range(title_index + 1, date_index)
            if position != company_index and _looks_like_location_line(scan_lines[position])
        ]
        _add_structured(
            scan_lines[title_index],
            company,
            line,
            ', '.join(location_parts)[:120] if location_parts else None,
        )

    # A following employer can omit the repeated role title. Reuse the most
    # recent teaching/faculty title, but only for a clearly named institution.
    for index, line in enumerate(scan_lines):
        if not DATE_RANGE_RE.search(line):
            continue
        following_company = (
            scan_lines[index + 1].strip()
            if index + 1 < len(scan_lines) else ''
        )
        if (
            _has_organization_hint(following_company)
            and _plausible_company(following_company)
            and not _is_probable_title(following_company)
        ):
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

    # A home-service entry may name the service and dates without stating a
    # formal position. Keep it transparently instead of borrowing a duty line
    # as the job title.
    for index in range(len(scan_lines) - 1):
        service = scan_lines[index].strip()
        date_line = scan_lines[index + 1].strip()
        if (
            re.fullmatch(r'(?i)tutorials?\s*\(\s*home\s+service\s*\)', service)
            and DATE_RANGE_RE.search(date_line)
        ):
            location = (
                scan_lines[index + 2].strip()
                if index + 2 < len(scan_lines)
                and _looks_like_location_line(scan_lines[index + 2])
                else None
            )
            _add_structured('Position Not Stated', service, date_line, location)

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
    # Keep high-confidence deterministic rows first, then allow the block
    # parser to contribute entries that use a different layout. Choosing only
    # one parser caused partial histories whenever the structured pass found
    # at least one (but not every) job.
    record_candidates = [*structured_records, *records]
    structured_count = len(structured_records)

    def employer_quality(item):
        company = (item.get('company') or '').casefold()
        score = 0
        if _has_organization_hint(company):
            score += 3
        if company == (item.get('job_title') or '').casefold():
            score -= 4
        if re.search(r'(?i)\b(?:activities|students|duties|summary|among|community)\b', company):
            score -= 3
        if company == 'not identified':
            score -= 2
        return score

    for candidate_index, rec in enumerate(record_candidates):
        if (not rec.get('job_title') or _is_duty_detail_line(rec.get('job_title')) or re.search(
            r'(?i)dean.?s\s+lister|president.?s\s+lister|with\s+honors|magna\s+cum\s+laude',
            rec.get('job_title') or ''
        )):
            continue
        if re.fullmatch(
            r'(?i)(?:officials?\s+and\s+faculty|faculty\s+and\s+officials?)',
            (rec.get('job_title') or '').strip(),
        ):
            continue
        if re.match(
            r'(?i)^teaching\b.*\b(?:subject|grade|during|semester|sem)\b',
            (rec.get('job_title') or '').strip(),
        ):
            continue
        company_value = (rec.get('company') or '').strip()
        if (
            not company_value
            or re.match(r'^[\W_]', company_value)
            or re.fullmatch(r'(?i)(?:n/?a|none|unknown)', company_value)
            or (
                _is_probable_title(company_value)
                and not re.search(
                    r'(?i)\b(?:university|school|academy|institute|center|centre|'
                    r'inc\.?|corporation|company|department|deped|foundation|authority)\b',
                    company_value,
                )
            )
        ):
            continue
        if any(
            existing['job_title'].casefold() == (rec.get('company') or '').casefold()
            and abs(float(existing.get('years') or 0) - float(rec.get('years') or 0)) < 0.05
            for existing in unique_records
        ):
            continue
        if company_value != 'Not Identified' and (
            not _plausible_company(company_value)
            or _looks_like_location_line(company_value)
        ):
            continue
        if (
            ',' in (rec.get('job_title') or '')
            and _has_organization_hint(rec.get('job_title'))
            and not _has_organization_hint(company_value)
        ):
            # A comma-separated ``Role, Employer`` row was reversed by the
            # fallback parser. The deterministic date-first pass already keeps
            # the correctly oriented record.
            continue
        location_value = (rec.get('location') or '').strip()
        if (
            location_value != 'Not Identified'
            and _has_organization_hint(location_value)
            and TITLE_KEYWORDS.search(f'{company_value} {location_value}')
            and not _has_organization_hint(company_value)
        ):
            # Another reversed fallback shape stores ``Role, Employer`` across
            # the company and location fields. Prefer the structured record.
            continue
        # A fallback block can span two neighboring jobs and produce a title
        # containing the employer from an already parsed structured row. Keep
        # the structured row and discard that cross-record combination.
        if candidate_index >= structured_count and any(
            existing.get('company')
            and existing['company'].casefold() in (rec.get('job_title') or '').casefold()
            and (
                existing['job_title'].casefold() in (rec.get('job_title') or '').casefold()
                or (rec.get('job_title') or '').casefold() in existing['job_title'].casefold()
            )
            for existing in structured_records
        ):
            continue
        is_valid, _ = validate_experience_record(rec)
        if not is_valid:
            continue
        key = (rec['job_title'].lower(), rec['company'].lower())
        overlapping_title = next((
            existing for existing in unique_records
            if abs(float(existing.get('years') or 0) - float(rec.get('years') or 0)) < 0.05
            and (rec.get('job_title') or '').casefold() in existing['job_title'].casefold()
            and employer_quality(existing) > employer_quality(rec)
        ), None)
        if overlapping_title is not None:
            continue
        same_role_index = next((
            index for index, existing in enumerate(unique_records)
            if existing['job_title'].casefold() == rec['job_title'].casefold()
            and abs(float(existing.get('years') or 0) - float(rec.get('years') or 0)) < 0.05
        ), None)
        if same_role_index is not None:
            existing_record = unique_records[same_role_index]
            existing_quality = employer_quality(existing_record)
            new_quality = employer_quality(rec)
            same_company = existing_record['company'].casefold() == rec['company'].casefold()
            if (not same_company and existing_quality >= 0 and new_quality >= 0
                    and existing_quality == new_quality):
                same_role_index = None
            elif (
                same_company
                and (existing_record.get('location') or 'Not Identified') == 'Not Identified'
                and (rec.get('location') or 'Not Identified') != 'Not Identified'
            ):
                old = unique_records[same_role_index]
                seen.discard((old['job_title'].lower(), old['company'].lower()))
                unique_records[same_role_index] = rec
                seen.add(key)
            elif new_quality > existing_quality:
                old = unique_records[same_role_index]
                seen.discard((old['job_title'].lower(), old['company'].lower()))
                unique_records[same_role_index] = rec
                seen.add(key)
            if same_role_index is not None:
                continue
        # A column-interleaved fallback may attach a nearby department label
        # to a role already captured with its dated employer. Treat an exact
        # title/duration pair as the same row when one employer is only a
        # generic organizational unit.
        generic_company = re.compile(
            r'(?i)^(?:senior\s+high\s+school|junior\s+high\s+school|elementary|'
            r'high\s+school|college|school)?\s*'
            r'(?:department|division|section|office|unit)$'
        )
        duplicate_title_index = next((
            index for index, existing in enumerate(unique_records)
            if existing['job_title'].casefold() == rec['job_title'].casefold()
            and abs(float(existing.get('years') or 0) - float(rec.get('years') or 0)) < 0.05
            and (generic_company.fullmatch(existing['company']) or generic_company.fullmatch(rec['company']))
        ), None)
        if duplicate_title_index is not None:
            if generic_company.fullmatch(unique_records[duplicate_title_index]['company']):
                old = unique_records[duplicate_title_index]
                seen.discard((old['job_title'].lower(), old['company'].lower()))
                unique_records[duplicate_title_index] = rec
                seen.add(key)
            continue
        overlapping_index = next((
            index for index, existing in enumerate(unique_records)
            if existing['company'].lower() == rec['company'].lower()
            and abs(float(existing.get('years') or 0) - float(rec.get('years') or 0)) < 0.05
            and (
                existing['job_title'].lower() in rec['job_title'].lower()
                or rec['job_title'].lower() in existing['job_title'].lower()
            )
        ), None)
        if overlapping_index is not None:
            if len(rec['job_title']) > len(unique_records[overlapping_index]['job_title']):
                old = unique_records[overlapping_index]
                seen.discard((old['job_title'].lower(), old['company'].lower()))
                unique_records[overlapping_index] = rec
                seen.add(key)
            continue
        if key not in seen:
            seen.add(key)
            unique_records.append(rec)
    return _drop_absorbed_records(unique_records)[:25]


def _drop_absorbed_records(records):
    """Remove duplicate roles whose title absorbed the employer line.

    Two of the block strategies can parse the same job differently: one yields
    "Principal Consultant" at "Bundok Industries", the other yields the whole
    header line "Bundok Industries Houston, TX - Principal Consultant" as the
    title and then takes a duty fragment as the employer. Both survive the
    key-based dedup because neither field matches. The absorbed variant is
    identifiable: same duration, and its title ends with the clean title.
    """
    def norm(value):
        return re.sub(r'[^a-z0-9]+', ' ', (value or '').lower()).strip()

    drop = set()
    for i, outer in enumerate(records):
        for j, inner in enumerate(records):
            if i == j or i in drop or j in drop:
                continue
            if float(outer.get('years') or 0) <= 0 or float(inner.get('years') or 0) <= 0:
                continue
            if round(outer.get('years') or 0, 2) != round(inner.get('years') or 0, 2):
                continue
            outer_title, inner_title = norm(outer.get('job_title')), norm(inner.get('job_title'))
            if not outer_title or not inner_title or outer_title == inner_title:
                continue
            # The longer title has swallowed the employer and location.
            outer_company = norm(outer.get('company'))
            inner_company = norm(inner.get('company'))
            same_company = bool(
                outer_company and outer_company == inner_company
            )
            swallowed_company = bool(
                inner_company and inner_company in outer_title
            )
            if (outer_title.endswith(inner_title)
                    and len(outer_title) > len(inner_title)
                    and (same_company or swallowed_company)):
                drop.add(i)
    return [rec for index, rec in enumerate(records) if index not in drop]
