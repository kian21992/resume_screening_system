from flask import Blueprint, render_template
from flask_login import login_required
from app.models import JobDescription, Applicant, Resume, ScreeningResult
from app.utils.device import current_device_id

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    device_id = current_device_id()
    total_jobs = JobDescription.query.count()
    total_applicants = Applicant.query.filter_by(device_id=device_id).count()
    total_resumes = Resume.query.filter_by(device_id=device_id).count()
    
    # Calculate pass rates based on recommendation labels
    qualified = ScreeningResult.query.filter_by(
        device_id=device_id,
        recommendation_label='Qualified',
    ).count()
    for_review = ScreeningResult.query.filter_by(
        device_id=device_id,
        recommendation_label='For Review',
    ).count()
    not_qualified = ScreeningResult.query.filter_by(
        device_id=device_id,
        recommendation_label='Not Qualified',
    ).count()
    
    stats = {
        'total_jobs': total_jobs,
        'total_applicants': total_applicants,
        'total_resumes': total_resumes,
        'qualified': qualified,
        'for_review': for_review,
        'not_qualified': not_qualified
    }
    
    # Get recent jobs
    recent_jobs = JobDescription.query.order_by(JobDescription.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html', stats=stats, recent_jobs=recent_jobs)
