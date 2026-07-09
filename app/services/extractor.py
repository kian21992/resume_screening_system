import os
import re

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
            page_text = page.extract_text(layout=False, x_tolerance=3, y_tolerance=3)
            if page_text:
                pages_text.append(page_text.strip())

            # Also pull any text locked inside table cells
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_cells = [cell.strip() for cell in row if cell and cell.strip()]
                    if row_cells:
                        pages_text.append("  ".join(row_cells))

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

    # Re-join soft-hyphenated line breaks: "Develop-\nment" → "Development"
    text = re.sub(r'(?<=[A-Za-z]{2})-\s*\n\s*(?=[A-Za-z]{2})', '', text)
    text = re.sub(r'\s+-\s*\n\s*', ' - ', text)
    text = re.sub(r'\n\s*-\s+', '\n- ', text)
    text = re.sub(r'\b((?:19|20)\d)\s+(\d)\b', r'\1\2', text)

    # Strip trailing spaces on each line
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
