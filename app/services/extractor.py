import os
import re


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
    for x, occupied in bins + [(end_x, 999)]:
        if occupied == 0 and run_start is None:
            run_start = x
        elif occupied != 0 and run_start is not None:
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
    columns = [
        [word for word in words if word['x1'] <= left_edge],
        [word for word in words if word['x0'] >= right_edge],
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
                        if normalized_row and normalized_row not in normalized_page:
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
    Tries pdfplumber first; falls back to PyPDF2 if the result is empty or errors.
    Also applies post-processing to fix common PDF artefacts:
      - Merges lines that were broken mid-word (soft hyphens)
      - Collapses excessive blank lines
      - Normalises whitespace
    """
    text = ""

    # Strategy 1: pdfplumber
    try:
        text = _extract_pdf_pdfplumber(filepath)
    except Exception as e:
        print(f"[extractor] pdfplumber failed ({e}), falling back to PyPDF2")

    # Strategy 2: PyPDF2 fallback
    if not text or len(text.strip()) < 50:
        try:
            text = _extract_pdf_pypdf2(filepath)
        except Exception as e:
            print(f"[extractor] PyPDF2 also failed: {e}")
            return None

    if not text or not text.strip():
        return None

    return _clean_extracted_text(text)


# ---------------------------------------------------------------------------
# DOCX extraction — reads paragraphs AND table cells so no content is missed
# ---------------------------------------------------------------------------

def extract_text_from_docx(filepath):
    """
    Extracts all text from a DOCX file including:
      - Regular paragraphs (body text, headings, bullet points)
      - Text inside table cells (skills matrices, experience tables, etc.)
      - Text boxes are not directly accessible via python-docx but the above
        covers the vast majority of resume content.
    """
    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = docx.Document(filepath)
    parts = []

    def _append_table(table):
        for row in table.rows:
            row_cells = []
            seen_cells = set()
            for cell in row.cells:
                cell_id = id(cell._tc)
                if cell_id in seen_cells:
                    continue
                seen_cells.add(cell_id)
                cell_text = cell.text.strip()
                if cell_text:
                    row_cells.append(cell_text)
            if row_cells:
                parts.append("  ".join(row_cells))

    # 1. Read header content first because resumes often place contact details there.
    seen_headers = set()
    for section in doc.sections:
        header_id = id(section.header._element)
        if header_id in seen_headers:
            continue
        seen_headers.add(header_id)
        for para in section.header.paragraphs:
            if para.text.strip():
                parts.append(para.text.strip())
        for table in section.header.tables:
            _append_table(table)

    # 2. Tables — iterate every cell in every row of every table
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, doc)
            if paragraph.text.strip():
                parts.append(paragraph.text.strip())
        elif isinstance(child, CT_Tbl):
            _append_table(Table(child, doc))

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
    text = "\n".join(line.strip() for line in text.splitlines())

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
