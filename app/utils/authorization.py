from functools import wraps

from flask import abort
from flask_login import current_user


VALID_ROLES = frozenset({'hr', 'manager', 'admin'})


def roles_required(*allowed_roles):
    """Reject authenticated users whose role is not explicitly allowed."""
    allowed = frozenset(allowed_roles)
    unknown = allowed - VALID_ROLES
    if unknown:
        raise ValueError(f'Unknown role(s): {", ".join(sorted(unknown))}')

    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if current_user.role not in allowed:
                abort(403)
            return view(*args, **kwargs)

        return wrapped

    return decorator
