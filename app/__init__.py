from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from config import Config
from sqlalchemy import inspect, text
from datetime import timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import os

db = SQLAlchemy()
login_manager = LoginManager()
bcrypt = Bcrypt()

def ensure_schema_columns():
    inspector = inspect(db.engine)
    table_columns = {
        table: {column['name'] for column in inspector.get_columns(table)}
        for table in inspector.get_table_names()
    }

    column_defs = {
        'jobs': {
            'critical_skills': 'TEXT',
        },
        'screening_results': {
            'matched_critical_skills': 'TEXT',
            'missing_critical_skills': 'TEXT',
            'decision_explanation': 'TEXT',
            'confidence_level': "VARCHAR(20) DEFAULT 'Medium'",
            'confidence_reason': 'TEXT',
            'screened_at': 'DATETIME',
            'reviewer_status': "VARCHAR(50) DEFAULT 'Pending Review'",
            'reviewer_notes': 'TEXT',
            'reviewed_by': 'INTEGER',
            'reviewed_at': 'DATETIME',
        },
        'resumes': {
            'uploaded_by': 'INTEGER',
        },
        'extracted_experience': {
            'location': 'VARCHAR(255)',
        },
    }

    for table, columns in column_defs.items():
        existing = table_columns.get(table, set())
        for column, column_type in columns.items():
            if column not in existing:
                db.session.execute(text(f'ALTER TABLE {table} ADD COLUMN {column} {column_type}'))
    db.session.commit()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)

    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'

    try:
        display_timezone = ZoneInfo(app.config.get('DISPLAY_TIMEZONE', 'Asia/Manila'))
    except ZoneInfoNotFoundError:
        display_timezone = timezone.utc

    @app.template_filter('local_datetime')
    def local_datetime(value, fmt='%b %d, %Y %I:%M %p'):
        if not value:
            return ''
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(display_timezone).strftime(fmt)

    @app.context_processor
    def inject_display_timezone():
        return {'display_timezone': app.config.get('DISPLAY_TIMEZONE', 'Asia/Manila')}

    with app.app_context():
        # Make sure uploads folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Import blueprints
        from app.routes.auth_routes import auth_bp
        from app.routes.dashboard_routes import dashboard_bp
        from app.routes.resume_routes import resume_bp
        from app.routes.job_routes import job_bp
        from app.routes.screening_routes import screening_bp

        # Register blueprints
        app.register_blueprint(auth_bp)
        app.register_blueprint(dashboard_bp)
        app.register_blueprint(resume_bp)
        app.register_blueprint(job_bp)
        app.register_blueprint(screening_bp)

        # Create all database tables
        db.create_all()
        ensure_schema_columns()

    return app
