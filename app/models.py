from app import db, login_manager
from flask_login import UserMixin
from datetime import datetime
import json

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='hr') # hr, manager, admin

class JobDescription(db.Model):
    __tablename__ = 'jobs'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    required_skills = db.Column(db.Text, nullable=False) # comma-separated
    critical_skills = db.Column(db.Text, nullable=True) # comma-separated must-have skills
    preferred_skills = db.Column(db.Text, nullable=True)
    education_req = db.Column(db.String(255), nullable=True)
    experience_req = db.Column(db.Integer, nullable=True) # years
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ScreeningCriteria(db.Model):
    __tablename__ = 'screening_criteria'
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
    min_fit_score = db.Column(db.Float, default=50.0)
    requires_all_critical = db.Column(db.Boolean, default=False)

class Applicant(db.Model):
    __tablename__ = 'applicants'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=True)
    phone = db.Column(db.String(50), nullable=True)
    applied_job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Resume(db.Model):
    __tablename__ = 'resumes'
    id = db.Column(db.Integer, primary_key=True)
    applicant_id = db.Column(db.Integer, db.ForeignKey('applicants.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    original_text = db.Column(db.Text(16777215), nullable=False) # MEDIUMTEXT
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    extraction_status = db.Column(db.String(50), default='pending') # pending, success, failed

    uploader = db.relationship('User', foreign_keys=[uploaded_by])

class ExtractedSkill(db.Model):
    __tablename__ = 'extracted_skills'
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=False)
    skill_name = db.Column(db.String(150), nullable=False)

class ExtractedEducation(db.Model):
    __tablename__ = 'extracted_education'
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=False)
    degree = db.Column(db.String(255), nullable=True)
    institution = db.Column(db.String(255), nullable=True)

class ExtractedExperience(db.Model):
    __tablename__ = 'extracted_experience'
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=False)
    job_title = db.Column(db.String(255), nullable=True)
    company = db.Column(db.String(255), nullable=True)
    location = db.Column(db.String(255), nullable=True)
    years = db.Column(db.Float, nullable=True)

class ExtractedCertification(db.Model):
    __tablename__ = 'extracted_certifications'
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=False)
    certification_name = db.Column(db.String(255), nullable=False)
    credential_type = db.Column(db.String(50), nullable=False, default='Certification')
    issuer = db.Column(db.String(255), nullable=True)
    date_obtained = db.Column(db.String(100), nullable=True)

class ScreeningResult(db.Model):
    __tablename__ = 'screening_results'
    REVIEW_STATUSES = (
        'Pending Review',
        'Reviewed',
        'Move to Interview',
        'Rejected',
        'Needs Clarification',
    )
    CONFIDENCE_LEVELS = ('High', 'Medium', 'Low')

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=False)
    applicant_id = db.Column(db.Integer, db.ForeignKey('applicants.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
    skill_score = db.Column(db.Float, default=0.0)
    experience_score = db.Column(db.Float, default=0.0)
    education_score = db.Column(db.Float, default=0.0)
    text_similarity_score = db.Column(db.Float, nullable=True)  # TF-IDF semantic similarity (0–100)
    fit_score = db.Column(db.Float, default=0.0)
    recommendation_label = db.Column(db.String(50), nullable=False) # Qualified, For Review, Not Qualified
    confidence_level = db.Column(db.String(20), nullable=False, default='Medium')
    confidence_reason = db.Column(db.Text, nullable=True)
    matched_skills = db.Column(db.Text, nullable=True) # JSON stored as text
    missing_skills = db.Column(db.Text, nullable=True) # JSON stored as text
    matched_critical_skills = db.Column(db.Text, nullable=True) # JSON stored as text
    missing_critical_skills = db.Column(db.Text, nullable=True) # JSON stored as text
    summary = db.Column(db.Text, nullable=True)
    decision_explanation = db.Column(db.Text, nullable=True)
    screened_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviewer_status = db.Column(db.String(50), nullable=False, default='Pending Review')
    reviewer_notes = db.Column(db.Text, nullable=True)
    reviewed_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)

    # Relationships
    applicant = db.relationship('Applicant', backref=db.backref('screening_results', lazy=True))
    job = db.relationship('JobDescription', backref=db.backref('screening_results', lazy=True))
    resume = db.relationship('Resume', backref=db.backref('screening_results', lazy=True))
    reviewer = db.relationship('User', foreign_keys=[reviewed_by])

    def set_matched_skills(self, skills):
        self.matched_skills = json.dumps(skills)

    def get_matched_skills(self):
        return json.loads(self.matched_skills) if self.matched_skills else []

    def set_missing_skills(self, skills):
        self.missing_skills = json.dumps(skills)

    def get_missing_skills(self):
        return json.loads(self.missing_skills) if self.missing_skills else []

    def set_matched_critical_skills(self, skills):
        self.matched_critical_skills = json.dumps(skills)

    def get_matched_critical_skills(self):
        return json.loads(self.matched_critical_skills) if self.matched_critical_skills else []

    def set_missing_critical_skills(self, skills):
        self.missing_critical_skills = json.dumps(skills)

    def get_missing_critical_skills(self):
        return json.loads(self.missing_critical_skills) if self.missing_critical_skills else []

class RecommendationLog(db.Model):
    __tablename__ = 'recommendation_logs'
    id = db.Column(db.Integer, primary_key=True)
    result_id = db.Column(db.Integer, db.ForeignKey('screening_results.id'), nullable=False)
    log_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
