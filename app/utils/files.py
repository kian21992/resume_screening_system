import os
import uuid

from werkzeug.utils import secure_filename


def unique_upload_filename(original_filename):
    """Return a safe filename with a unique prefix to avoid overwrites."""
    safe_name = secure_filename(original_filename or "resume")
    stem, ext = os.path.splitext(safe_name)
    stem = stem or "resume"
    return f"{stem}_{uuid.uuid4().hex[:12]}{ext.lower()}"


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
