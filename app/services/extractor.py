import os
import re
import logging


logger = logging.getLogger(__name__)


_SECTION_HEADING_RE = re.compile(
    r"(?im)^\s*(?:professional\s+summary|summary|profile|education|"
    r"(?:professional|work(?:ing)?|teaching|employment)\s+experience|experience|"
    r"technical\s+skills|skills(?:\s*(?:&|and)\s*qualities)?|"
    r"certifications?|certificates?|licenses?|projects?|contact|"
    r"personal\s+(?:information|data)|scholastic\s+records?)\s*:?[ \t]*$"
)


def _text_quality_score(text):
    """Estimate whether extracted resume text is readable and well structured."""
    if not text or not text.strip():
        return float('-inf')

    cleaned = _clean_extracted_text(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", cleaned)
    if not words:
        return float('-inf')

    visible_chars = [char for char in cleaned if not char.isspace()]
    alpha_ratio = (
        sum(char.isalpha() for char in visible_chars) / len(visible_chars)
        if visible_chars else 0
    )
    section_count = len(_SECTION_HEADING_RE.findall(cleaned))
    contact_count = int(bool(re.search(r'[\w.+-]+@[\w.-]+\.\w{2,}', cleaned)))
    contact_count += int(bool(re.search(r'(?:\+?\d[\d ()-]{7,}\d)', cleaned)))
    date_count = len(re.findall(r'\b(?:19|20)\d{2}\b', cleaned))
    private_glyphs = len(re.findall(r'[\uE000-\uF8FF\uFFFD]', cleaned))
    flattened_chars = sum(max(0, len(line) - 220) for line in lines)
    concatenated_headings = len(re.findall(
        r'\b(?:WORKEXPERIENCE|PROFESSIONALSUMMARY|TECHNICALSKILLS|'
        r'PERSONALINFORMATION|EDUCATIONALBACKGROUND)\b',
        cleaned,
        flags=re.IGNORECASE,
    ))
    punctuation_spacing = len(re.findall(r'\s+[,.!?;:]', cleaned))
    one_word_line_ratio = (
        sum(len(line.split()) == 1 for line in lines) / len(lines)
        if lines else 0
    )
    # Some PDFs expose every positioned word as a separate text line. That
    # fragmentation can falsely increase both the line-count reward and the
    # section-heading count (a lone ``Education`` word looks like a heading),
    # even though downstream parsers can no longer reconstruct phrases. Keep
    # normal short bullet lists untouched and penalize only large documents
    # where single-word lines dominate.
    word_line_fragmentation = (
        max(0.0, one_word_line_ratio - 0.55) * 180
        if len(lines) >= 40 else 0.0
    )
    # A recognized heading at the very end has no section content beneath it.
    # This commonly indicates that a coordinate-unaware PDF reader moved a
    # two-column heading after the entries it labels. Treat that as a strong
    # reading-order defect even when the candidate contains slightly more text.
    trailing_orphaned_heading = int(bool(
        lines and _SECTION_HEADING_RE.fullmatch(lines[-1])
    ))

    return (
        min(len(words), 1200) * 0.035
        + min(len(lines), 120) * 0.18
        + section_count * 5
        + contact_count * 2.5
        + min(date_count, 20) * 0.35
        + alpha_ratio * 8
        - private_glyphs * 1.5
        - flattened_chars * 0.035
        - concatenated_headings * 5
        - punctuation_spacing * 0.12
        - trailing_orphaned_heading * 8
        - word_line_fragmentation
    )


def _select_best_pdf_text(candidates):
    """Return the highest-quality non-empty extraction and its method name."""
    usable = [
        (method, text, _text_quality_score(text))
        for method, text in candidates
        if text and text.strip()
    ]
    if not usable:
        return None, None

    method, text, _score = max(usable, key=lambda item: item[2])
    return text, method


def _words_to_lines(words, y_tolerance=3):
    """Rebuild readable lines from positioned PDF words."""
    rows = []
    for word in sorted(words, key=lambda item: (item['top'], item['x0'])):
        row = next(
            (candidate for candidate in reversed(rows[-3:])
             if abs(candidate['top'] - word['top']) <= y_tolerance),
            None,
        )
        if row is None:
            row = {'top': word['top'], 'words': []}
            rows.append(row)
        row['words'].append(word)

    lines = []
    for row in sorted(rows, key=lambda item: item['top']):
        ordered = sorted(row['words'], key=lambda item: item['x0'])
        parts = []
        previous_x1 = None
        for word in ordered:
            if previous_x1 is not None:
                parts.append('  ' if word['x0'] - previous_x1 >= 12 else ' ')
            parts.append(word['text'])
            previous_x1 = word['x1']
        line = ''.join(parts).strip()
        if line:
            lines.append(line)
    return lines


def _find_column_gutter(words, page_width):
    """Return a conservative two-column gutter, or None for normal pages."""
    if len(words) < 35 or page_width <= 0:
        return None

    bin_width = 2.0
    start_x = page_width * 0.22
    end_x = page_width * 0.68
    bins = []
    x = start_x
    while x < end_x:
        occupied = sum(1 for word in words if word['x0'] < x + bin_width and word['x1'] > x)
        bins.append((x, occupied))
        x += bin_width

    runs = []
    run_start = None
    # Allow one crossing word: decorative rules, headings, and URLs can cross
    # an otherwise stable gutter on template-based resumes.
    for x, occupied in bins + [(end_x, 999)]:
        if occupied <= 1 and run_start is None:
            run_start = x
        elif occupied > 1 and run_start is not None:
            if x - run_start >= 16:
                runs.append((run_start, x))
            run_start = None
    if not runs:
        return None

    for left_edge, right_edge in sorted(runs, key=lambda run: run[1] - run[0], reverse=True):
        left_words = [word for word in words if word['x1'] <= left_edge]
        right_words = [word for word in words if word['x0'] >= right_edge]
        # Require meaningful content on both sides so margins and decorative
        # whitespace do not trigger column mode.
        if len(left_words) >= 12 and len(right_words) >= 20:
            smaller_share = min(len(left_words), len(right_words)) / len(words)
            if smaller_share >= 0.12:
                return left_edge, right_edge
    return None


def _extract_page_columns(page):
    """Extract a true two-column page one column at a time."""
    words = page.extract_words(x_tolerance=1, y_tolerance=3, keep_blank_chars=False)
    gutter = _find_column_gutter(words, page.width)
    if not gutter:
        return None

    left_edge, right_edge = gutter
    split_x = (left_edge + right_edge) / 2
    # Assign gutter-crossing words by their center instead of dropping them.
    # Large names and headings often extend slightly into the whitespace.
    columns = [
        [word for word in words if (word['x0'] + word['x1']) / 2 < split_x],
        [word for word in words if (word['x0'] + word['x1']) / 2 >= split_x],
    ]
    columns = [column for column in columns if column]
    columns.sort(key=lambda column: min(word['top'] for word in column))
    return '\n'.join(
        line
        for column in columns
        for line in _words_to_lines(column)
    )

# ---------------------------------------------------------------------------
# PDF extraction — tries pdfplumber first (much more accurate for resumes),
# then falls back to PyPDF2 if pdfplumber fails or yields nothing.
# ---------------------------------------------------------------------------

def _extract_pdf_pdfplumber(filepath):
    """
    Primary PDF extractor using pdfplumber.
    - Handles multi-column layouts and tables better than PyPDF2
    - Preserves spacing between words that PyPDF2 often concatenates
    - Reads text in reading order (top→bottom, left→right per page)
    """
    import pdfplumber

    pages_text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            # extract_text with layout=True tries to preserve visual order
            # A small x tolerance is important for tightly kerned/template-based
            # resumes; larger values concatenate words ("DevelopLessonPlans").
            page_text = _extract_page_columns(page)
            if not page_text:
                page_text = page.extract_text(layout=False, x_tolerance=1, y_tolerance=3)
            page_text = page_text.strip() if page_text else ""
            if page_text:
                pages_text.append(page_text)

            # Also pull any text locked inside table cells
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_cells = [cell.strip() for cell in row if cell and cell.strip()]
                    if row_cells:
                        # pdfplumber's normal text extraction usually already
                        # contains table text. Add a row only when it is truly
                        # absent; blindly appending every table duplicated resume
                        # entries and made adjacent fields appear related.
                        row_text = "  ".join(row_cells)
                        normalized_row = re.sub(r'\s+', ' ', row_text).strip().lower()
                        normalized_page = re.sub(r'\s+', ' ', page_text).lower()
                        # On template-based resumes, pdfplumber can identify the
                        # entire page as a table even though extract_text already
                        # returned every cell. Appending partially matching rows
                        # then duplicates most of the resume and destroys section
                        # boundaries. Table fallback is only needed for pages whose
                        # ordinary extraction is sparse.
                        page_is_sparse = len(normalized_page) < 200
                        if (page_is_sparse and normalized_row
                                and normalized_row not in normalized_page):
                            pages_text.append(row_text)

    return "\n".join(pages_text)


def _extract_pdf_pypdf2(filepath):
    """
    Fallback PDF extractor using PyPDF2.
    Uses visitor-based extraction when available for better character grouping.
    """
    import PyPDF2

    parts = []
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text.strip())
    return "\n".join(parts)


def extract_text_from_pdf(filepath):
    """
    Runs both supported PDF parsers and keeps the more readable result.
    Also applies post-processing to fix common PDF artefacts:
      - Merges lines that were broken mid-word (soft hyphens)
      - Collapses excessive blank lines
      - Normalises whitespace
    """
    candidates = []
    for method, extractor in (
        ('pdfplumber', _extract_pdf_pdfplumber),
        ('PyPDF2', _extract_pdf_pypdf2),
    ):
        try:
            candidates.append((method, extractor(filepath)))
        except Exception as exc:
            logger.warning('%s extraction failed for %s: %s', method, filepath, exc)

    text, method = _select_best_pdf_text(candidates)
    if not text:
        return None

    logger.debug('Selected %s extraction for %s', method, filepath)

    return _clean_extracted_text(text)


# ---------------------------------------------------------------------------
# DOCX extraction — reads paragraphs AND table cells so no content is missed
# ---------------------------------------------------------------------------

def extract_text_from_docx(filepath):
    """
    Extracts all text from a DOCX file including:
      - Regular paragraphs (body text, headings, bullet points)
      - Text inside table cells (skills matrices, experience tables, etc.)
      - Header and footer content
      - Text stored in DrawingML/VML text boxes
    """
    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = docx.Document(filepath)
    parts = []

    def _textbox_texts(element):
        """Recover text that python-docx omits from Paragraph.text."""
        values = []
        for container in element.xpath('.//*[local-name()="txbxContent"]'):
            text = ' '.join(
                node.text.strip()
                for node in container.xpath('.//*[local-name()="t"]')
                if node.text and node.text.strip()
            )
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                values.append(text)
        return values

    def _append_paragraph(paragraph):
        paragraph_text = paragraph.text.strip()
        if paragraph_text:
            parts.append(paragraph_text)
        for textbox_text in _textbox_texts(paragraph._p):
            if textbox_text != paragraph_text:
                parts.append(textbox_text)

    def _append_table(table):
        rows = []
        for row in table.rows:
            row_cells = []
            seen_cells = set()
            for cell in row.cells:
                cell_id = id(cell._tc)
                if cell_id in seen_cells:
                    continue
                seen_cells.add(cell_id)
                cell_parts = []
                cell_text = cell.text.strip()
                if cell_text:
                    cell_parts.append(cell_text)
                for textbox_text in _textbox_texts(cell._tc):
                    if textbox_text not in cell_parts:
                        cell_parts.append(textbox_text)
                if cell_parts:
                    row_cells.append('\n'.join(cell_parts))
            if row_cells:
                rows.append(row_cells)

        if not rows:
            return
        header = [re.sub(r'\s+', ' ', cell).strip().casefold() for cell in rows[0]]
        all_labels = {
            re.sub(r'\s+', ' ', cell).strip().casefold()
            for row in rows for cell in row
        }

        # Preserve the meaning of common application-form resume tables rather
        # than flattening headers, values, and references into ambiguous prose.
        if {'full name', 'residence'}.issubset(all_labels) and any(
            'email' in value or 'cellphone' in value for value in all_labels
        ):
            for row in rows:
                for index in range(0, len(row) - 1, 2):
                    label, value = row[index:index + 2]
                    if label.strip() and value.strip():
                        parts.append(f'{label.strip()}: {value.strip()}')
            return

        if 'level' in header and any('school name' in value for value in header):
            parts.append('EDUCATION')
            for row in rows[1:]:
                padded = row + [''] * (5 - len(row))
                level, school, course, location, year = padded[:5]
                if not level.strip() or not school.strip():
                    continue
                credential = course.strip() if course.strip().casefold() not in {'', 'n/a', 'na'} else level.strip()
                if credential.casefold() == 'elementary':
                    credential = 'Elementary Education'
                parts.extend((credential, school.strip()))
            return

        if any('company' in value or 'organization' in value for value in header) and 'position' in header:
            parts.append('WORK EXPERIENCE')
            for row in rows[1:]:
                padded = row + [''] * (6 - len(row))
                company, position, location, start, end, duties = padded[:6]
                if not company.strip() and not position.strip():
                    continue
                date_range = f'{start.strip()} - {end.strip()}'.strip(' -')
                parts.append(' | '.join(value for value in (
                    position.strip() or 'Position Not Stated', company.strip(),
                    location.strip(), date_range,
                ) if value))
                if duties.strip():
                    parts.append(f'Duties: {duties.strip()}')
            return

        if 'full name' in header and any('relationship' in value for value in header):
            parts.append('CHARACTER REFERENCES')
            for row in rows[1:]:
                parts.append(' | '.join(value.strip() for value in row if value.strip()))
            return

        for row in rows:
            parts.append("  ".join(row))

    # 1. Read header content first because resumes often place contact details there.
    def _append_story(story):
        for child in story._element.iterchildren():
            if isinstance(child, CT_P):
                _append_paragraph(Paragraph(child, story))
            elif isinstance(child, CT_Tbl):
                _append_table(Table(child, story))

    seen_headers = set()
    for section in doc.sections:
        for header in (
            section.header,
            section.first_page_header,
            section.even_page_header,
        ):
            header_id = id(header._element)
            if header_id in seen_headers:
                continue
            seen_headers.add(header_id)
            _append_story(header)

    # 2. Tables — iterate every cell in every row of every table
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            _append_paragraph(Paragraph(child, doc))
        elif isinstance(child, CT_Tbl):
            _append_table(Table(child, doc))

    seen_footers = set()
    for section in doc.sections:
        for footer in (
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        ):
            footer_id = id(footer._element)
            if footer_id in seen_footers:
                continue
            seen_footers.add(footer_id)
            _append_story(footer)

    text = "\n".join(parts)
    return _clean_extracted_text(text) if text.strip() else None


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

def _clean_extracted_text(text):
    """
    Fixes common PDF/DOCX extraction artefacts without losing meaningful content:
      - Re-joins words broken across lines with a soft hyphen (e.g. 'Develop-\nment')
      - Collapses 3+ consecutive blank lines into 2 (preserve section breaks)
      - Normalises non-breaking spaces and other Unicode whitespace to regular spaces
      - Strips leading/trailing whitespace per line
    """
    if not text:
        return text

    # Normalise Unicode whitespace characters to regular spaces/newlines
    text = text.replace('\xa0', ' ')   # non-breaking space
    text = text.replace('\u2019', "'") # right single quotation mark
    text = text.replace('\u2018', "'") # left single quotation mark
    text = text.replace('\u2013', '-') # en dash
    text = text.replace('\u2014', '-') # em dash
    text = text.replace('\u2022', '-') # bullet •
    text = text.replace('\uf0b7', '-') # Wingdings bullet
    text = text.replace('\uf06c', '-') # common private-font bullet
    text = text.replace('\u200b', '')  # zero-width space
    text = text.replace('\u00ad', '')  # soft hyphen

    # Re-join soft-hyphenated line breaks: "Develop-\nment" → "Development"
    text = re.sub(r'(?<=[A-Za-z]{2})-\s*\n\s*(?=[A-Za-z]{2})', '', text)
    text = re.sub(r'\s+-\s*\n\s*', ' - ', text)
    text = re.sub(r'\n\s*-\s+', '\n- ', text)
    text = re.sub(r'\b((?:19|20)\d)\s+(\d)\b', r'\1\2', text)

    # Strip trailing spaces on each line
    # Preserve meaningful multi-space/tab column separators; the structured
    # experience parser uses them to distinguish company and role columns.
    def normalize_decorative_heading(line):
        """Collapse all-caps headings exported as one glyph per word.

        Canva and similar resume builders often encode ``WORK EXPERIENCE`` as
        ``W O R K  E X P E R I E N C E``. Only known headings are rewritten,
        avoiding unsafe guesses about spacing in letter-spaced names.
        """
        tokens = line.strip().split()
        if len(tokens) < 4:
            return line.strip()
        glyph_share = sum(
            len(re.sub(r'[^A-Za-z]', '', token)) <= 1 for token in tokens
        ) / len(tokens)
        if glyph_share < 0.75:
            return line.strip()
        compact = re.sub(r'[^A-Za-z&]', '', line).upper()
        headings = {
            'PROFESSIONALPROFILE': 'PROFESSIONAL PROFILE',
            'CONTACT': 'CONTACT',
            'EDUCATION': 'EDUCATION',
            'SKILLS': 'SKILLS',
            'LANGUAGE': 'LANGUAGE',
            'LANGUAGES': 'LANGUAGES',
            'WORKEXPERIENCE': 'WORK EXPERIENCE',
            'PERSONALINFORMATION': 'PERSONAL INFORMATION',
            'CHARACTERREFERENCES': 'CHARACTER REFERENCES',
            'PUBLICATIONS&PRESENTATIONS': 'PUBLICATIONS & PRESENTATIONS',
            'PROFESSIONAL&CIVICAFFILIATION': 'PROFESSIONAL & CIVIC AFFILIATION',
            'LICENSES&CERTIFICATIONS': 'LICENSES & CERTIFICATIONS',
        }
        return headings.get(compact, line.strip())

    cleaned_lines = [normalize_decorative_heading(line) for line in text.splitlines()]

    # Layered PDF text and duplicated DOCX table content can emit the same
    # line multiple times in succession. Remove only adjacent normalized
    # duplicates; repeated headings or employers elsewhere remain meaningful.
    deduplicated_lines = []
    previous_key = None
    for line in cleaned_lines:
        key = re.sub(r'[^\w]+', ' ', line, flags=re.UNICODE).strip().casefold()
        if key and key == previous_key:
            continue
        deduplicated_lines.append(line)
        previous_key = key if key else None
    text = "\n".join(deduplicated_lines)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_text_from_file(filepath):
    """
    Extracts text from a PDF or DOCX file.
    Returns the full extracted text string, or None if extraction fails.
    """
    if not os.path.exists(filepath):
        print(f"[extractor] File not found: {filepath}")
        return None

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext == '.docx':
        return extract_text_from_docx(filepath)
    else:
        print(f"[extractor] Unsupported file format: {ext}")
        return None
