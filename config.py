import os
import secrets

from dotenv import load_dotenv


# Load local settings for every entry point, including ``python app.py`` and
# maintenance scripts that import create_app directly.
load_dotenv()


def resolve_secret_key(environment, configured_secret):
    """Require a strong configured key in production; generate one for local use."""
    environment = (environment or 'development').strip().lower()
    configured_secret = (configured_secret or '').strip()
    unsafe_example_secrets = {
        'change-me',
        'dev-only-change-before-deployment',
        'replace-with-a-random-secret-of-at-least-32-characters',
    }

    if environment in {'production', 'prod'} and (
        len(configured_secret) < 32 or configured_secret in unsafe_example_secrets
    ):
        raise RuntimeError(
            'SECRET_KEY must be set to at least 32 characters when APP_ENV=production.'
        )

    return configured_secret or secrets.token_urlsafe(32)

class Config:
    APP_ENV = os.environ.get('APP_ENV', 'development').strip().lower()
    SECRET_KEY = resolve_secret_key(APP_ENV, os.environ.get('SECRET_KEY'))
    # SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'mysql+pymysql://root:@localhost/resume_screening_db'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///resume_screening.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'instance', 'uploads'
    )
    DISPLAY_TIMEZONE = os.environ.get('DISPLAY_TIMEZONE') or 'Asia/Manila'
    MAX_CONTENT_LENGTH = 160 * 1024 * 1024  # up to 10 resumes per upload batch
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE = APP_ENV in {'production', 'prod'}
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = 'Lax'
    REMEMBER_COOKIE_SECURE = APP_ENV in {'production', 'prod'}
    RATELIMIT_STORAGE_URI = os.environ.get('RATELIMIT_STORAGE_URI', 'memory://')
    RATELIMIT_HEADERS_ENABLED = True
