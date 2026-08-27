import os

from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from app import db
from app.models import (
    JobDescription, ScreeningCriteria, Applicant, Resume,
    ScreeningResult, RecommendationLog, ExtractedSkill,
    ExtractedEducation, ExtractedExperience, ExtractedCertification
)
from app.utils.files import safe_delete_uploaded_file
from app.utils.authorization import roles_required
from app.utils.device import current_device_id

job_bp = Blueprint('job', __name__)

def legacy_upload_folder():
    return os.path.join(current_app.root_path, 'static', 'uploads')

def owned_job_or_404(job_id):
    return JobDescription.query.filter_by(
        id=job_id,
        device_id=current_device_id(),
    ).first_or_404()

@job_bp.route('/jobs')
@login_required
def list_jobs():
    jobs = JobDescription.query.filter_by(
        device_id=current_device_id(),
    ).order_by(JobDescription.created_at.desc()).all()
    return render_template('jobs/list.html', jobs=jobs)

@job_bp.route('/jobs/create', methods=['GET', 'POST'])
@login_required
@roles_required('manager', 'admin')
def create_job():
    device_id = current_device_id()
    if request.method == 'POST':
        title = request.form.get('title')
        required_skills = request.form.get('required_skills')
        critical_skills = request.form.get('critical_skills', '')
        preferred_skills = request.form.get('preferred_skills', '')
        education_req = request.form.get('education_req', '')
        experience_req = request.form.get('experience_req', 0, type=int)
        
        # Criteria fields
        min_fit_score = request.form.get('min_fit_score', 50.0, type=float)
        requires_all_critical = request.form.get('requires_all_critical') == 'on'
        
        if not title or not required_skills:
            flash('Title and Required Skills are mandatory.', 'danger')
            return redirect(url_for('job.create_job'))
            
        new_job = JobDescription(
            device_id=device_id,
            title=title,
            required_skills=required_skills,
            critical_skills=critical_skills,
            preferred_skills=preferred_skills,
            education_req=education_req,
            experience_req=experience_req,
            created_by=current_user.id
        )
        
        db.session.add(new_job)
        db.session.flush() # get new_job.id
        
        criteria = ScreeningCriteria(
            device_id=device_id,
            job_id=new_job.id,
            min_fit_score=min_fit_score,
            requires_all_critical=requires_all_critical
        )
        db.session.add(criteria)
        db.session.commit()
        
        flash('Job Listing created successfully!', 'success')
        return redirect(url_for('job.list_jobs'))
        
    return render_template('jobs/create.html')

@job_bp.route('/jobs/<int:job_id>/edit', methods=['GET', 'POST'])
@login_required
@roles_required('manager', 'admin')
def edit_job(job_id):
    device_id = current_device_id()
    job = owned_job_or_404(job_id)
    criteria = ScreeningCriteria.query.filter_by(
        job_id=job.id,
        device_id=device_id,
    ).first()

    if request.method == 'POST':
        title = request.form.get('title')
        required_skills = request.form.get('required_skills')
        critical_skills = request.form.get('critical_skills', '')
        preferred_skills = request.form.get('preferred_skills', '')
        education_req = request.form.get('education_req', '')
        experience_req = request.form.get('experience_req', 0, type=int)
        min_fit_score = request.form.get('min_fit_score', 50.0, type=float)
        requires_all_critical = request.form.get('requires_all_critical') == 'on'

        if not title or not required_skills:
            flash('Title and Required Skills are mandatory.', 'danger')
            return redirect(url_for('job.edit_job', job_id=job_id))

        job.title = title
        job.required_skills = required_skills
        job.critical_skills = critical_skills
        job.preferred_skills = preferred_skills
        job.education_req = education_req
        job.experience_req = experience_req

        if criteria:
            criteria.min_fit_score = min_fit_score
            criteria.requires_all_critical = requires_all_critical
        else:
            criteria = ScreeningCriteria(
                device_id=device_id,
                job_id=job.id,
                min_fit_score=min_fit_score,
                requires_all_critical=requires_all_critical
            )
            db.session.add(criteria)

        db.session.commit()
        flash('Job listing updated successfully!', 'success')
        return redirect(url_for('job.view_job', job_id=job_id))

    return render_template('jobs/edit.html', job=job, criteria=criteria)

@job_bp.route('/jobs/<int:job_id>')
@login_required
def view_job(job_id):
    device_id = current_device_id()
    job = owned_job_or_404(job_id)
    criteria = ScreeningCriteria.query.filter_by(
        job_id=job.id,
        device_id=device_id,
    ).first()
    return render_template('jobs/view.html', job=job, criteria=criteria)

@job_bp.route('/jobs/<int:job_id>/delete', methods=['POST'])
@login_required
@roles_required('admin')
def delete_job(job_id):
    device_id = current_device_id()
    job = owned_job_or_404(job_id)

    # A historical shared deployment can leave another device's candidate rows
    # linked to this job. Preserve them instead of crossing the ownership
    # boundary or violating their foreign keys.
    other_device_data_exists = (
        Applicant.query.filter(
            Applicant.applied_job_id == job.id,
            Applicant.device_id != device_id,
        ).first() is not None
        or Resume.query.filter(
            Resume.job_id == job.id,
            Resume.device_id != device_id,
        ).first() is not None
        or ScreeningResult.query.filter(
            ScreeningResult.job_id == job.id,
            ScreeningResult.device_id != device_id,
        ).first() is not None
    )
    if other_device_data_exists:
        flash(
            'This job cannot be deleted while protected legacy candidate records exist.',
            'danger',
        )
        return redirect(url_for('job.view_job', job_id=job.id))
    
    # 1. Get resumes to remove physical files and delete from extracted fields
    resumes = Resume.query.filter_by(job_id=job.id, device_id=device_id).all()
    resume_ids = [r.id for r in resumes]
    
    # 2. Get screening results to delete logs
    results = ScreeningResult.query.filter_by(job_id=job.id, device_id=device_id).all()
    result_ids = [res.id for res in results]
    
    # 3. Delete from recommendation_logs
    if result_ids:
        RecommendationLog.query.filter(
            RecommendationLog.result_id.in_(result_ids),
            RecommendationLog.device_id == device_id,
        ).delete(synchronize_session=False)
        
    # 4. Delete from screening_results
    ScreeningResult.query.filter_by(
        job_id=job.id,
        device_id=device_id,
    ).delete(synchronize_session=False)
    
    # 5. Delete from extracted_skills, extracted_education, extracted_experience
    if resume_ids:
        ExtractedSkill.query.filter(ExtractedSkill.resume_id.in_(resume_ids), ExtractedSkill.device_id == device_id).delete(synchronize_session=False)
        ExtractedEducation.query.filter(ExtractedEducation.resume_id.in_(resume_ids), ExtractedEducation.device_id == device_id).delete(synchronize_session=False)
        ExtractedExperience.query.filter(ExtractedExperience.resume_id.in_(resume_ids), ExtractedExperience.device_id == device_id).delete(synchronize_session=False)
        ExtractedCertification.query.filter(ExtractedCertification.resume_id.in_(resume_ids), ExtractedCertification.device_id == device_id).delete(synchronize_session=False)
        
    # 6. Delete physical resume files from disk
    for r in resumes:
        if r.filepath:
            try:
                safe_delete_uploaded_file(
                    r.filepath,
                    current_app.config['UPLOAD_FOLDER'],
                    legacy_upload_folder()
                )
            except Exception:
                pass
                
    # 7. Delete from resumes
    Resume.query.filter_by(job_id=job.id, device_id=device_id).delete(synchronize_session=False)
    
    # 8. Delete from applicants
    Applicant.query.filter_by(
        applied_job_id=job.id,
        device_id=device_id,
    ).delete(synchronize_session=False)
    
    # 9. Delete from screening_criteria
    ScreeningCriteria.query.filter_by(
        job_id=job.id,
        device_id=device_id,
    ).delete(synchronize_session=False)
    
    # 10. Delete the job itself
    db.session.delete(job)
    db.session.commit()
    
    flash(f'Job "{job.title}" and all associated candidate records have been successfully deleted.', 'success')
    return redirect(url_for('job.list_jobs'))
