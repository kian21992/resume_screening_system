import os
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import current_user, login_required
from app import db
from app.models import (
    JobDescription, ScreeningResult, Applicant, Resume,
    ExtractedEducation, ExtractedExperience, ExtractedSkill, ExtractedCertification,
    RecommendationLog
)
from app.services.evidence import build_candidate_evidence
from app.services.recommender import analyze_preferred_skills, extract_resume_skills
from app.utils.files import safe_delete_uploaded_file
from app.utils.authorization import roles_required

screening_bp = Blueprint('screening', __name__)

def legacy_upload_folder():
    return os.path.join(current_app.root_path, 'static', 'uploads')

def _screening_filter_args():
    job_id = request.args.get('job_id', type=int)
    reviewer_status = (request.args.get('reviewer_status') or '').strip()
    if reviewer_status not in ScreeningResult.REVIEW_STATUSES:
        reviewer_status = ''
    return job_id, reviewer_status

def _apply_screening_filters(query, job_id=None, reviewer_status=''):
    if job_id:
        query = query.filter_by(job_id=job_id)
    if reviewer_status:
        query = query.filter_by(reviewer_status=reviewer_status)
    return query

def _screening_sort_arg():
    sort = (request.args.get('sort') or 'fit_desc').strip()
    return sort if sort in {'fit_desc', 'fit_asc', 'newest', 'oldest'} else 'fit_desc'

def _screening_order_columns(sort):
    return {
        'fit_desc': (ScreeningResult.fit_score.desc(), ScreeningResult.id.desc()),
        'fit_asc': (ScreeningResult.fit_score.asc(), ScreeningResult.id.desc()),
        'newest': (ScreeningResult.screened_at.desc(), ScreeningResult.id.desc()),
        'oldest': (ScreeningResult.screened_at.asc(), ScreeningResult.id.asc()),
    }[sort]

def _ordered_screening_results_query(job_id=None, reviewer_status='', sort='fit_desc'):
    query = _apply_screening_filters(ScreeningResult.query, job_id, reviewer_status)
    return query.order_by(*_screening_order_columns(sort))

def _candidate_navigation(result_id, job_id=None, reviewer_status='', sort='fit_desc'):
    query = _ordered_screening_results_query(job_id, reviewer_status, sort)
    ordered_results = query.all()
    result_ids = [item.id for item in ordered_results]

    if result_id not in result_ids:
        return None, None, None, len(result_ids)

    index = result_ids.index(result_id)
    previous_result = ordered_results[index - 1] if index > 0 else None
    next_result = ordered_results[index + 1] if index < len(ordered_results) - 1 else None
    return previous_result, next_result, index + 1, len(result_ids)

@screening_bp.route('/screening_results')
@login_required
def screening_results():
    job_id, reviewer_status = _screening_filter_args()
    sort = _screening_sort_arg()
    jobs = JobDescription.query.all()

    query = _ordered_screening_results_query(job_id, reviewer_status, sort)
    results = query.all()
        
    return render_template(
        'screening/results.html',
        jobs=jobs,
        results=results,
        selected_job_id=job_id,
        selected_reviewer_status=reviewer_status,
        selected_sort=sort,
        reviewer_statuses=ScreeningResult.REVIEW_STATUSES,
    )

@screening_bp.route('/screening_results/<int:result_id>')
@login_required
def result_detail(result_id):
    job_id, reviewer_status = _screening_filter_args()
    sort = _screening_sort_arg()
    result = ScreeningResult.query.get_or_404(result_id)
    applicant = Applicant.query.get(result.applicant_id)
    job = JobDescription.query.get(result.job_id)
    resume = Resume.query.get(result.resume_id)
    previous_result, next_result, candidate_position, candidate_total = _candidate_navigation(
        result.id,
        job_id,
        reviewer_status,
        sort,
    )
    
    # Query extracted metadata
    education = ExtractedEducation.query.filter_by(resume_id=resume.id).all()
    experience = ExtractedExperience.query.filter_by(resume_id=resume.id).all()
    certifications = ExtractedCertification.query.filter_by(resume_id=resume.id).all()
    stored_skills = ExtractedSkill.query.filter_by(resume_id=resume.id).all()
    preferred_skills = [
        skill.strip() for skill in (job.preferred_skills or '').split(',')
        if skill.strip()
    ]
    matched_preferred, _, _ = analyze_preferred_skills(
        resume.original_text, preferred_skills
    )
    configured_skills = [
        skill.strip()
        for value in (job.required_skills, job.critical_skills, job.preferred_skills)
        for skill in (value or '').split(',')
        if skill.strip()
    ]
    # Re-extract on display so results created before this feature also show a
    # complete list. Stored rows remain the source for newly uploaded resumes.
    live_extracted_skills = extract_resume_skills(resume.original_text, configured_skills)
    # Live extraction lets older screening records benefit from algorithm
    # improvements. Stored rows are a fallback for an empty/unreadable section.
    extracted_skills = live_extracted_skills or [
        item.skill_name for item in stored_skills
    ]
    candidate_evidence = build_candidate_evidence(
        resume_text=resume.original_text,
        matched_skills=result.get_matched_skills(),
        missing_skills=result.get_missing_skills(),
        matched_critical_skills=result.get_matched_critical_skills(),
        missing_critical_skills=result.get_missing_critical_skills(),
        matched_preferred_skills=matched_preferred,
        experience_records=experience,
        education_records=education,
        experience_requirement=job.experience_req or 0,
        education_requirement=job.education_req,
    )
    
    return render_template('screening/detail.html', 
                           result=result, 
                           applicant=applicant, 
                           job=job, 
                           resume=resume,
                           education=education,
                           experience=experience,
                           certifications=certifications,
                           extracted_skills=extracted_skills,
                           candidate_evidence=candidate_evidence,
                           previous_result=previous_result,
                           next_result=next_result,
                           candidate_position=candidate_position,
                           candidate_total=candidate_total,
                           selected_job_id=job_id,
                           selected_reviewer_status=reviewer_status,
                           selected_sort=sort)


@screening_bp.route('/screening_results/<int:result_id>/review', methods=['POST'])
@login_required
@roles_required('hr', 'manager', 'admin')
def update_review(result_id):
    result = ScreeningResult.query.get_or_404(result_id)
    reviewer_status = (request.form.get('reviewer_status') or '').strip()
    reviewer_notes = (request.form.get('reviewer_notes') or '').strip()

    if reviewer_status not in ScreeningResult.REVIEW_STATUSES:
        flash('Invalid reviewer status.', 'danger')
        return redirect(url_for('screening.result_detail', result_id=result.id))
    if len(reviewer_notes) > 3000:
        flash('Reviewer notes must be 3,000 characters or fewer.', 'danger')
        return redirect(url_for('screening.result_detail', result_id=result.id))

    result.reviewer_status = reviewer_status
    result.reviewer_notes = reviewer_notes or None
    result.reviewed_by = current_user.id
    result.reviewed_at = datetime.utcnow()

    log_text = f'Human review updated by {current_user.username}: {reviewer_status}.'
    if reviewer_notes:
        log_text += f' Notes: {reviewer_notes}'
    db.session.add(RecommendationLog(result_id=result.id, log_text=log_text))
    db.session.commit()

    flash('Reviewer decision and notes saved.', 'success')
    return redirect(url_for(
        'screening.result_detail',
        result_id=result.id,
        job_id=request.args.get('job_id', type=int),
        reviewer_status=request.args.get('reviewer_status') or None,
    ))

@screening_bp.route('/screening_results/<int:result_id>/delete', methods=['POST'])
@login_required
@roles_required('admin')
def delete_candidate(result_id):
    result = ScreeningResult.query.get_or_404(result_id)
    job_id = result.job_id
    applicant_id = result.applicant_id
    resume_id = result.resume_id
    
    # 1. Delete recommendation logs
    RecommendationLog.query.filter_by(result_id=result_id).delete(synchronize_session=False)
    
    # 2. Delete physical resume file
    resume = Resume.query.get(resume_id)
    if resume:
        if resume.filepath:
            try:
                safe_delete_uploaded_file(
                    resume.filepath,
                    current_app.config['UPLOAD_FOLDER'],
                    legacy_upload_folder()
                )
            except Exception:
                pass
                
        # 3. Delete extracted details
        ExtractedSkill.query.filter_by(resume_id=resume_id).delete(synchronize_session=False)
        ExtractedEducation.query.filter_by(resume_id=resume_id).delete(synchronize_session=False)
        ExtractedExperience.query.filter_by(resume_id=resume_id).delete(synchronize_session=False)
        ExtractedCertification.query.filter_by(resume_id=resume_id).delete(synchronize_session=False)
        
        # 4. Delete Resume DB record
        db.session.delete(resume)
        
    # 5. Delete ScreeningResult DB record
    db.session.delete(result)
    
    # 6. Delete Applicant DB record
    applicant = Applicant.query.get(applicant_id)
    if applicant:
        db.session.delete(applicant)
        
    db.session.commit()
    flash('Candidate record successfully removed.', 'success')
    return redirect(url_for(
        'screening.screening_results',
        job_id=request.args.get('job_id', type=int) or job_id,
        reviewer_status=request.args.get('reviewer_status') or None,
    ))

@screening_bp.route('/screening_results/delete_all', methods=['POST'])
@login_required
@roles_required('admin')
def delete_all_candidates():
    job_id = request.args.get('job_id', type=int)
    reviewer_status = (request.args.get('reviewer_status') or '').strip()
    if reviewer_status not in ScreeningResult.REVIEW_STATUSES:
        reviewer_status = ''
    
    # Query targets
    result_query = ScreeningResult.query
    if job_id:
        result_query = result_query.filter_by(job_id=job_id)
    if reviewer_status:
        result_query = result_query.filter_by(reviewer_status=reviewer_status)

    results = result_query.all()
    resume_ids = [res.resume_id for res in results]
    applicant_ids = [res.applicant_id for res in results]
    result_ids = [res.id for res in results]

    resumes = Resume.query.filter(Resume.id.in_(resume_ids)).all() if resume_ids else []
    applicants = Applicant.query.filter(Applicant.id.in_(applicant_ids)).all() if applicant_ids else []
        
    # 1. Delete logs
    if result_ids:
        RecommendationLog.query.filter(RecommendationLog.result_id.in_(result_ids)).delete(synchronize_session=False)
        
    # 2. Delete screening results
    if result_ids:
        ScreeningResult.query.filter(ScreeningResult.id.in_(result_ids)).delete(synchronize_session=False)
        
    # 3. Delete extracted features
    if resume_ids:
        ExtractedSkill.query.filter(ExtractedSkill.resume_id.in_(resume_ids)).delete(synchronize_session=False)
        ExtractedEducation.query.filter(ExtractedEducation.resume_id.in_(resume_ids)).delete(synchronize_session=False)
        ExtractedExperience.query.filter(ExtractedExperience.resume_id.in_(resume_ids)).delete(synchronize_session=False)
        ExtractedCertification.query.filter(ExtractedCertification.resume_id.in_(resume_ids)).delete(synchronize_session=False)
        
    # 4. Delete files
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
                
    # 5. Delete resumes & applicants
    if resume_ids:
        Resume.query.filter(Resume.id.in_(resume_ids)).delete(synchronize_session=False)
    if applicant_ids:
        Applicant.query.filter(Applicant.id.in_(applicant_ids)).delete(synchronize_session=False)
        
    db.session.commit()
    
    if reviewer_status and job_id:
        flash(f'All {reviewer_status} candidates for this job listing have been removed.', 'success')
    elif reviewer_status:
        flash(f'All {reviewer_status} candidates have been removed.', 'success')
    elif job_id:
        flash('All candidates for this job listing have been successfully removed.', 'success')
    else:
        flash('All candidates in the system have been successfully removed.', 'success')
        
    return redirect(url_for(
        'screening.screening_results',
        job_id=job_id,
        reviewer_status=reviewer_status or None,
    ))
