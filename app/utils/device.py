import re
import secrets

from flask import session


DEVICE_SESSION_KEY = 'device_id'
DEVICE_ID_LENGTH = 64
LEGACY_DEVICE_ID = '0' * DEVICE_ID_LENGTH
_DEVICE_ID_PATTERN = re.compile(r'^[0-9a-f]{64}$')


def is_valid_device_id(value):
    """Return whether ``value`` is a usable browser/device identifier."""
    return bool(
        isinstance(value, str)
        and value != LEGACY_DEVICE_ID
        and _DEVICE_ID_PATTERN.fullmatch(value)
    )


def current_device_id():
    """Get or create the persistent random ID stored in the signed session."""
    device_id = session.get(DEVICE_SESSION_KEY)
    if not is_valid_device_id(device_id):
        device_id = secrets.token_hex(DEVICE_ID_LENGTH // 2)
        session[DEVICE_SESSION_KEY] = device_id

    # Flask gives permanent sessions an expiry date. The configured lifetime is
    # refreshed as the browser continues to use the application.
    session.permanent = True
    return device_id
