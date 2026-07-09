import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-only-change-before-deployment'
    # SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'mysql+pymysql://root:@localhost/resume_screening_db'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///resume_screening.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'instance', 'uploads'
    )
    DISPLAY_TIMEZONE = os.environ.get('DISPLAY_TIMEZONE') or 'Asia/Manila'
    MAX_CONTENT_LENGTH = 160 * 1024 * 1024  # up to 10 resumes per upload batch
