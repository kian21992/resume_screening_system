import os
import re
import logging


logger = logging.getLogger(__name__)


_SECTION_HEADING_RE = re.compile(
    r"(?im)^\s*(?:professional\s+summary|summary|profile|about\s+me|career\s+objectives?|"
    r"education(?:al)?(?:\s+(?:background|history|attainment))?|"
    r"(?:professional|work(?:ing)?|teaching|employment)\s+experience|experience(?:/s)?|"
    r"work\s+(?:and|&)\s+training\s+experience|technical\s+skills|"
    r"qualifications?\s+(?:and|&)\s+skills|skills(?:\s*(?:&|and)\s+qualities)?|"
    r"certifications?|certificates?|licenses?|projects?|contact|"
    r"trainings?|seminars?|trainings?\s*(?:&|and|/)\s*seminars?|"
    r"seminars?\s*(?:&|and|/)\s*trainings?|character\s+references?|"
    r"references?|personal\s+(?:information|data|profile)|scholastic\s+records?|"
    r"achievements?|awards?|languages?|interests?\s+(?:and|&)\s+hobbies)\s*:?[ \t]*$"
)


_POSITIONED_HEADING_KEYS = {
    'ACHIEVEMENTS', 'AWARDS', 'CAREEROBJECTIVE', 'CAREEROBJECTIVES',
    'CHARACTERREFERENCE', 'CHARACTERREFERENCES', 'CONTACT',
    'CONTACTINFORMATION', 'EDUCATION', 'EDUCATIONALATTAINMENT',
    'EDUCATIONALBACKGROUND', 'EDUCATIONALHISTORY', 'EMPLOYMENTHISTORY',
    'EXPERIENCE', 'EXPERIENCES', 'INTERESTSANDHOBBIES', 'LANGUAGE',
    'LANGUAGES', 'PERSONALINFORMATION', 'PERSONALPROFILE',
    'PROFESSIONALEXPERIENCE', 'QUALIFICATIONSANDSKILLS', 'REFERENCE',
    'REFERENCES', 'SEMINAR', 'SEMINARS', 'SKILL', 'SKILLS', 'TRAINING',
    'TRAININGS', 'WORKANDTRAININGEXPERIENCE', 'WORKEXPERIENCE',
}


def _heading_key(value):
    """Return a punctuation/spacing-insensitive key for known headings."""
    return re.sub(r'[^A-Za-z]', '', value or '').upper()


def _looks_like_positioned_heading(value):
    return _heading_key(value) in _POSITIONED_HEADING_KEYS


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
    # Some visually ordered PDFs store each text box as a separate object and
    # expose several section headings before any of their content. A run of
    # headings near the start is therefore strong evidence that the candidate
    # follows content-stream order instead of the page's reading order.
    early_heading_run = 0
    current_heading_run = 0
    for line in lines[:10]:
        if _SECTION_HEADING_RE.fullmatch(line):
            current_heading_run += 1
            early_heading_run = max(early_heading_run, current_heading_run)
        else:
            current_heading_run = 0
    orphaned_heading_run = max(0, early_heading_run - 1)
    first_heading_index = next(
        (index for index, line in enumerate(lines) if _SECTION_HEADING_RE.fullmatch(line)),
        None,
    )
    contact_near_top = any(
        re.search(r'[\w.+-]+@[\w.-]+\.\w{2,}|(?:\+?\d[\d ()-]{7,}\d)', line)
        for line in lines[:12]
    )
    late_first_heading = (
        max(0, first_heading_index - 16)
        if contact_near_top and first_heading_index is not None else 0
    )
    standalone_punctuation_lines = sum(
        bool(re.fullmatch(r'[:|/.-]+', line)) for line in lines
    )
    # PyPDF2 occasionally inserts a space before the final glyph of common
    # resume words.  A coordinate extraction without these artifacts is safer
    # when the overall quality scores are otherwise nearly tied.
    split_word_artifacts = len(re.findall(
        r'(?i)\b(?:experienc\s+e|teache\s+r|advise\s+r|technolog\s+y|'
        r'accountanc\s+y|performa\s+nce|managemen\s+t|educatio\s+n)\b',
        cleaned,
    ))
    longest_one_word_run = 0
    one_word_run = 0
    for line in lines:
        if len(re.findall(r"[A-Za-z][A-Za-z'\-]*", line)) == 1:
            one_word_run += 1
            longest_one_word_run = max(longest_one_word_run, one_word_run)
        else:
            one_word_run = 0
    # A long uninterrupted run is a stronger defect than the overall ratio:
    # legitimate skill lists may contain many one-word items, but normally
    # not 8+ fragments from a single sentence.
    fragmented_sentence_run = max(0, longest_one_word_run - 6)

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
        - orphaned_heading_run * 5
        - word_line_fragmentation
        - min(late_first_heading, 30) * 0.8
        - max(0, standalone_punctuation_lines - 2) * 1.5
        - split_word_artifacts * 2
        - fragmented_sentence_run * 1.25
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


def _words_to_rows(words, y_tolerance=3):
    """Group positioned PDF words into visual rows."""
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

    return sorted(rows, key=lambda item: item['top'])


def _row_to_line(words):
    ordered = sorted(words, key=lambda item: item['x0'])
    parts = []
    previous_x1 = None
    for word in ordered:
        if previous_x1 is not None:
            parts.append('  ' if word['x0'] - previous_x1 >= 12 else ' ')
        parts.append(word['text'])
        previous_x1 = word['x1']
    return ''.join(parts).strip()


def _words_to_lines(words, y_tolerance=3):
    """Rebuild readable lines from positioned PDF words."""
    rows = _words_to_rows(words, y_tolerance=y_tolerance)

    lines = []
    for row in rows:
        line = _row_to_line(row['words'])
        if line:
            lines.append(line)
    return lines


def _find_paired_heading_split(words, page_width):
    """Find a split supported by two independent headings on the same row.

    This is intentionally stricter than a whitespace-only gutter.  It enables
    mixed-layout pages (full-width header, two-column middle, full-width body)
    without turning ordinary label/value tables into columns.
    """
    if len(words) < 45 or page_width <= 0:
        return None

    rows = _words_to_rows(words)
    best = None
    step = max(4.0, page_width * 0.012)
    split = page_width * 0.34
    while split <= page_width * 0.66:
        left_count = sum((word['x0'] + word['x1']) / 2 < split for word in words)
        right_count = len(words) - left_count
        if min(left_count, right_count) < 18:
            split += step
            continue

        left_heading_rows = []
        right_heading_rows = []
        paired_rows = []
        heading_gaps = []
        for index, row in enumerate(rows):
            left = [word for word in row['words'] if (word['x0'] + word['x1']) / 2 < split]
            right = [word for word in row['words'] if (word['x0'] + word['x1']) / 2 >= split]
            left_text = _row_to_line(left)
            right_text = _row_to_line(right)
            left_heading = _looks_like_positioned_heading(left_text)
            right_heading = _looks_like_positioned_heading(right_text)
            if left_heading:
                left_heading_rows.append((index, row['top'], left))
            if right_heading:
                right_heading_rows.append((index, row['top'], right))
            if left_heading and right_heading:
                paired_rows.append(index)
                heading_gaps.append((max(word['x1'] for word in left), min(word['x0'] for word in right)))

        # Template columns often place their colored heading boxes a few
        # points above/below one another. Treat nearby opposing headings as a
        # pair, but require multiple such pairs when they are not on one row.
        adjacent_pairs = []
        for left_index, left_top, left_words in left_heading_rows:
            nearest = min(
                right_heading_rows,
                key=lambda item: abs(item[1] - left_top),
                default=None,
            )
            if nearest and abs(nearest[1] - left_top) <= 24:
                right_index, _right_top, right_words = nearest
                adjacent_pairs.append((left_index, right_index))
                heading_gaps.append((
                    max(word['x1'] for word in left_words),
                    min(word['x0'] for word in right_words),
                ))

        if not paired_rows and len(adjacent_pairs) < 2:
            split += step
            continue
        paired_rows.extend(index for pair in adjacent_pairs for index in pair)
        paired_rows = sorted(set(paired_rows))

        if paired_rows:
            balance = min(left_count, right_count) / max(left_count, right_count)
            valid_gaps = [
                (left_edge, right_edge)
                for left_edge, right_edge in heading_gaps
                if right_edge - left_edge >= 12
            ]
            if not valid_gaps:
                split += step
                continue
            gap_midpoints = sorted((left + right) / 2 for left, right in valid_gaps)
            derived_split = gap_midpoints[len(gap_midpoints) // 2]
            candidate = (
                len(paired_rows), balance, -abs(derived_split - page_width / 2),
                derived_split, paired_rows,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        split += step
    return (best[3], best[4]) if best else None


def _extract_mixed_page_regions(words, page_width):
    """Read only the proven two-column region column-by-column."""
    detected = _find_paired_heading_split(words, page_width)
    if not detected:
        return None
    split_x, paired_rows = detected
    rows = _words_to_rows(words)
    start_index = min(paired_rows)
    end_index = len(rows)

    # Once the first paired headings establish where the mixed column region
    # begins, whitespace in that region is better evidence for the actual
    # boundary than heading widths (a short ``SKILLS`` heading may end far
    # before the rest of its column). This also avoids cutting off words near
    # the inner edge of an uneven pair of columns.
    preliminary_region = [
        word for row in rows[start_index:] for word in row['words']
    ]
    regional_gutter = _find_column_gutter(preliminary_region, page_width)
    if regional_gutter:
        split_x = sum(regional_gutter) / 2

    def fragments(row):
        left = [word for word in row['words'] if (word['x0'] + word['x1']) / 2 < split_x]
        right = [word for word in row['words'] if (word['x0'] + word['x1']) / 2 >= split_x]
        return left, right

    # A later single heading followed by continuous full-width prose closes the
    # column region.  Independent content on both sides keeps the region open.
    for index in range(start_index + 2, len(rows)):
        left, right = fragments(rows[index])
        left_text, right_text = _row_to_line(left), _row_to_line(right)
        single_heading = (
            (_looks_like_positioned_heading(left_text) and not right_text)
            or (_looks_like_positioned_heading(right_text) and not left_text)
        )
        if not single_heading:
            continue
        continuous_rows = 0
        independent_rows = 0
        for future in rows[index + 1:index + 9]:
            future_left, future_right = fragments(future)
            if not future_left or not future_right:
                continue
            gap = min(word['x0'] for word in future_right) - max(word['x1'] for word in future_left)
            if gap >= 10:
                independent_rows += 1
            else:
                continuous_rows += 1
        if continuous_rows >= 2 and continuous_rows > independent_rows:
            end_index = index
            break

    before_words = [word for row in rows[:start_index] for word in row['words']]
    region_words = [word for row in rows[start_index:end_index] for word in row['words']]
    after_words = [word for row in rows[end_index:] for word in row['words']]
    left_region = [word for word in region_words if (word['x0'] + word['x1']) / 2 < split_x]
    right_region = [word for word in region_words if (word['x0'] + word['x1']) / 2 >= split_x]

    parts = []
    for group in (before_words, left_region, right_region, after_words):
        group_text = '\n'.join(_words_to_lines(group))
        if group_text.strip():
            parts.append(group_text)
    return '\n'.join(parts) if parts else None


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
    for left_edge, right_edge in sorted(runs, key=lambda run: run[1] - run[0], reverse=True):
        left_words = [word for word in words if word['x1'] <= left_edge]
        right_words = [word for word in words if word['x0'] >= right_edge]
        # Require meaningful content on both sides so margins and decorative
        # whitespace do not trigger column mode.
        if len(left_words) >= 12 and len(right_words) >= 20:
            smaller_share = min(len(left_words), len(right_words)) / len(words)
            if smaller_share >= 0.12:
                return left_edge, right_edge

    # Large names, colored heading bands, and lines that reach the edge of a
    # column can cross an otherwise valid gutter. In those templates the word
    # *centers* still form two clear clusters. Use that evidence only for a
    # large central gap with substantial content on both sides; this keeps
    # ordinary single-column resumes and right-aligned date rows in one flow.
    centers = sorted({
        round((word['x0'] + word['x1']) / 2, 2)
        for word in words
        if page_width * 0.12 <= (word['x0'] + word['x1']) / 2 <= page_width * 0.88
    })
    minimum_center_gap = max(20.0, page_width * 0.045)
    center_gaps = sorted(
        (
            (right - left, left, right)
            for left, right in zip(centers, centers[1:])
            if right - left >= minimum_center_gap
        ),
        reverse=True,
    )
    for _gap, left_edge, right_edge in center_gaps:
        split_x = (left_edge + right_edge) / 2
        left_words = [
            word for word in words
            if (word['x0'] + word['x1']) / 2 < split_x
        ]
        right_words = [
            word for word in words
            if (word['x0'] + word['x1']) / 2 >= split_x
        ]
        if len(left_words) < 20 or len(right_words) < 20:
            continue
        smaller_share = min(len(left_words), len(right_words)) / len(words)
        if smaller_share >= 0.20:
            return left_edge, right_edge
    return None


def _extract_page_columns(page):
    """Extract a true two-column page one column at a time."""
    words = page.extract_words(x_tolerance=1, y_tolerance=3, keep_blank_chars=False)
    gutter = _find_column_gutter(words, page.width)
    if not gutter:
        return _extract_mixed_page_regions(words, page.width)

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

    # A portrait can delay the first text in an identity sidebar, causing a
    # body section on the right to be read before the applicant's name.  Change
    # the established ordering only when the left side contains contact
    # evidence and the earlier right side begins with an actual section.
    if len(columns) == 2:
        physical_left, physical_right = sorted(
            columns, key=lambda column: min(word['x0'] for word in column)
        )
        left_text = '\n'.join(_words_to_lines(physical_left))
        right_lines = _words_to_lines(physical_right)
        right_first = right_lines[0] if right_lines else ''
        left_has_contact = bool(re.search(
            r'[\w.+-]+@[\w.-]+\.\w{2,}|(?:\+?\d[\d ()-]{7,}\d)',
            left_text,
        ))
        if left_has_contact and _looks_like_positioned_heading(right_first):
            columns = [physical_left, physical_right]
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
        # python-docx exposes a vertically merged cell again through every row
        # that the cell spans. Track the underlying XML elements for the whole
        # table so a resume section stored in one tall layout cell is emitted
        # once rather than repeated after each neighboring sidebar row.
        seen_cells = set()
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                cell_element = cell._tc
                if cell_element in seen_cells:
                    continue
                seen_cells.add(cell_element)
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
    # Join genuine word wrapping (``Develop-`` / ``ment``), but preserve a
    # capitalized next line. In compact resumes ``Coordinator-`` followed by
    # ``National University`` is a role/company separator, not hyphenation.
    text = re.sub(r'(?<=[A-Za-z]{2})-\s*\n\s*(?=[a-z]{2})', '', text)
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
        raw = line.strip()
        tokens = raw.split()
        compact = _heading_key(raw)
        headings = {
            'ACHIEVEMENTS': 'ACHIEVEMENTS',
            'CAREEROBJECTIVE': 'CAREER OBJECTIVE',
            'CAREEROBJECTIVES': 'CAREER OBJECTIVE',
            'CHARACTERREFERENCE': 'CHARACTER REFERENCE',
            'CHARACTERREFERENCES': 'CHARACTER REFERENCES',
            'CONTACT': 'CONTACT',
            'CONTACTINFORMATION': 'CONTACT INFORMATION',
            'EDUCATION': 'EDUCATION',
            'EDUCATIONALATTAINMENT': 'EDUCATIONAL ATTAINMENT',
            'EDUCATIONALBACKGROUND': 'EDUCATIONAL BACKGROUND',
            'EDUCATIONALHISTORY': 'EDUCATIONAL HISTORY',
            'EMPLOYMENTHISTORY': 'EMPLOYMENT HISTORY',
            'EXPERIENCES': 'EXPERIENCE',
            'INTERESTSANDHOBBIES': 'INTERESTS AND HOBBIES',
            'LANGUAGE': 'LANGUAGE',
            'LANGUAGES': 'LANGUAGES',
            'LICENSEDPROFESSIONALTEACHER': 'LICENSED PROFESSIONAL TEACHER',
            'PERSONALINFORMATION': 'PERSONAL INFORMATION',
            'PERSONALPROFILE': 'PERSONAL PROFILE',
            'PROFESSIONALCIVICAFFILIATION': 'PROFESSIONAL & CIVIC AFFILIATION',
            'PROFESSIONALPROFILE': 'PROFESSIONAL PROFILE',
            'PROFESSIONALEXPERIENCE': 'PROFESSIONAL EXPERIENCE',
            'PUBLICATIONSPRESENTATIONS': 'PUBLICATIONS & PRESENTATIONS',
            'QUALIFICATIONSANDSKILLS': 'QUALIFICATIONS AND SKILLS',
            'REFERENCE': 'REFERENCE',
            'REFERENCES': 'REFERENCES',
            'SEMINAR': 'SEMINAR',
            'SEMINARS': 'SEMINARS',
            'SKILLS': 'SKILLS',
            'TRAINING': 'TRAINING',
            'TRAININGS': 'TRAININGS',
            'WORKANDTRAININGEXPERIENCE': 'WORK AND TRAINING EXPERIENCE',
            'WORKEXPERIENCE': 'WORK EXPERIENCE',
            'LICENSESCERTIFICATIONS': 'LICENSES & CERTIFICATIONS',
        }
        glyph_share = sum(
            len(re.sub(r'[^A-Za-z]', '', token)) <= 1 for token in tokens
        ) / len(tokens) if tokens else 0
        decorative_spacing = len(tokens) >= 4 and glyph_share >= 0.75
        internal_split_keys = {
            'CONTACTINFORMATION',
            'EDUCATIONALATTAINMENT',
            'PERSONALINFORMATION',
        }
        internal_word_split = bool(
            compact in headings
            and compact in internal_split_keys
            and len(tokens) >= 2
            and any(
                len(re.sub(r'[^A-Za-z]', '', token)) == 1
                for token in tokens
            )
            and raw.upper() == raw
        )
        # Repair letter-spaced headings and short internal word splits such as
        # ``PERSONAL I NFORMATION``. Ordinary headings retain their original
        # spelling, case, and punctuation for backward-compatible text.
        if compact in headings and (decorative_spacing or internal_word_split):
            return headings[compact]
        return raw

    cleaned_lines = [normalize_decorative_heading(line) for line in text.splitlines()]

    # A narrow sidebar can wrap a letter-spaced professional credential over
    # several physical rows. Join only this exact known phrase; arbitrary line
    # joining could change valid names or section content.
    credential_key = 'LICENSEDPROFESSIONALTEACHER'
    joined_lines = []
    index = 0
    while index < len(cleaned_lines):
        compact = _heading_key(cleaned_lines[index])
        matched_end = None
        if compact and credential_key.startswith(compact):
            combined = compact
            for end in range(index + 1, min(index + 4, len(cleaned_lines))):
                combined += _heading_key(cleaned_lines[end])
                if combined == credential_key:
                    matched_end = end
                    break
                if not credential_key.startswith(combined):
                    break
        if matched_end is not None:
            joined_lines.append('LICENSED PROFESSIONAL TEACHER')
            index = matched_end + 1
            continue
        joined_lines.append(cleaned_lines[index])
        index += 1
    cleaned_lines = joined_lines

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
