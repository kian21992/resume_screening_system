import os
import uuid
from datetime import datetime

from werkzeug.utils import secure_filename


def unique_upload_filename(original_filename):
    """Return a safe filename with a unique prefix to avoid overwrites."""
    safe_name = secure_filename(original_filename or "resume")
    stem, ext = os.path.splitext(safe_name)
    stem = stem or "resume"
    return f"{stem}_{uuid.uuid4().hex[:12]}{ext.lower()}"


def job_upload_directory(upload_root, job_id, job_title, uploaded_at=None):
    """Return the job/year/month/week directory for a resume upload."""
    uploaded_at = uploaded_at or datetime.now()
    safe_title = secure_filename(job_title or "job").lower().replace("_", "-")
    safe_title = safe_title or "job"
    job_folder = f"{int(job_id)}-{safe_title}"
    month_folder = uploaded_at.strftime("%m-%B").lower()
    week_of_month = ((uploaded_at.day - 1) // 7) + 1

    return os.path.join(
        os.path.abspath(upload_root),
        "jobs",
        job_folder,
        str(uploaded_at.year),
        month_folder,
        f"week-{week_of_month}",
    )


def is_path_inside_directory(path, directory):
    path = os.path.abspath(path)
    directory = os.path.abspath(directory)
    try:
        return os.path.commonpath([path, directory]) == directory
    except ValueError:
        return False


def safe_delete_uploaded_file(filepath, *allowed_upload_folders):
    """Delete an uploaded file only if it lives inside an allowed upload folder."""
    if not filepath:
        return False

    abs_path = os.path.abspath(filepath)
    allowed = any(
        is_path_inside_directory(abs_path, folder)
        for folder in allowed_upload_folders
        if folder
    )
    if not allowed:
        return False

    if os.path.exists(abs_path):
        os.remove(abs_path)
        return True

    return False
