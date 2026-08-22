from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFError, CSRFProtect
import click
from config import Config
from sqlalchemy import inspect, text
from datetime import timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import os

db = SQLAlchemy()
login_manager = LoginManager()
bcrypt = Bcrypt()
csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address)

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
    csrf.init_app(app)
    limiter.init_app(app)

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

    @app.errorhandler(CSRFError)
    def handle_csrf_error(error):
        return render_template(
            'errors/http_error.html',
            code=400,
            title='Invalid or expired form',
            message='Please go back, refresh the page, and submit the form again.',
        ), 400

    @app.errorhandler(403)
    def forbidden(_error):
        return render_template(
            'errors/http_error.html',
            code=403,
            title='Permission denied',
            message='Your account is not authorized to perform this action.',
        ), 403

    @app.errorhandler(429)
    def too_many_requests(_error):
        return render_template(
            'errors/http_error.html',
            code=429,
            title='Too many attempts',
            message='Please wait before trying to sign in again.',
        ), 429

    @app.cli.command('set-user-role')
    @click.argument('username')
    @click.argument('role', type=click.Choice(['hr', 'manager', 'admin']))
    def set_user_role(username, role):
        """Assign one of the application's approved roles to an existing user."""
        from app.models import User

        user = User.query.filter_by(username=username).first()
        if user is None:
            raise click.ClickException(f'User "{username}" was not found.')

        user.role = role
        db.session.commit()
        click.echo(f'Updated {username} to role: {role}')

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
