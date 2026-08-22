import os
import re
import hashlib
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import current_user, login_required
from app import db
from app.models import (
    JobDescription, Applicant, Resume, ScreeningResult, RecommendationLog,
    ScreeningCriteria, ExtractedSkill, ExtractedEducation, ExtractedExperience,
    ExtractedCertification
)
from app.services.extractor import extract_text_from_file
from app.services.recommender import evaluate_candidate
from app.utils.files import (
    job_upload_directory,
    safe_delete_uploaded_file,
    unique_upload_filename,
)
from app.utils.authorization import roles_required

resume_bp = Blueprint('resume', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_RESUMES_PER_SCREENING = 10

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def legacy_upload_folder():
    return os.path.join(current_app.root_path, 'static', 'uploads')

def _clean_identity_value(value):
    value = (value or '').strip()
    if not value or value.lower().startswith('unknown'):
        return ''
    return value.lower()

def _normalize_duplicate_text(value):
    value = re.sub(r'[^a-z0-9]+', ' ', (value or '').lower())
    return re.sub(r'\s+', ' ', value).strip()

def _normalize_phone(value):
    return re.sub(r'\D+', '', value or '')

def _file_sha256(filepath):
    digest = hashlib.sha256()
    with open(filepath, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _find_duplicate_file(saved_filepath):
    if not os.path.exists(saved_filepath):
        return None

    uploaded_hash = _file_sha256(saved_filepath)
    for existing_resume in Resume.query.all():
        if not existing_resume.filepath or not os.path.exists(existing_resume.filepath):
            continue
        if os.path.abspath(existing_resume.filepath) == os.path.abspath(saved_filepath):
            continue
        if _file_sha256(existing_resume.filepath) == uploaded_hash:
            return existing_resume

    return None

def _experience_signature(records):
    signature = []
    for record in records or []:
        if isinstance(record, dict):
            job_title = record.get('job_title')
            company = record.get('company')
            years = record.get('years')
        else:
            job_title = record.job_title
            company = record.company
            years = record.years
        signature.append((
            _normalize_duplicate_text(job_title),
            _normalize_duplicate_text(company),
            round(float(years or 0), 2),
        ))
    return tuple(signature)

def _education_signature(records):
    signature = []
    for record in records or []:
        if isinstance(record, dict):
            degree = record.get('degree')
            institution = record.get('institution')
        else:
            degree = record.degree
            institution = record.institution
        signature.append((
            _normalize_duplicate_text(degree),
            _normalize_duplicate_text(institution),
        ))
    return tuple(signature)

def _resume_text_signature(text):
    return _normalize_duplicate_text(text)

def _find_duplicate_resume(extracted_text, evaluation):
    contact_info = evaluation.get('contact_info') or {}
    email = _clean_identity_value(contact_info.get('email'))
    phone = _normalize_phone(contact_info.get('phone'))
    name = _clean_identity_value(contact_info.get('name'))
    exp_sig = _experience_signature(evaluation.get('extracted_exp'))
    edu_sig = _education_signature(evaluation.get('extracted_edu'))
    text_sig = _resume_text_signature(extracted_text)

    for existing_resume in Resume.query.all():
        existing_applicant = Applicant.query.get(existing_resume.applicant_id)
        if not existing_applicant:
            continue

        existing_email = _clean_identity_value(existing_applicant.email)
        existing_phone = _normalize_phone(existing_applicant.phone)
        existing_name = _clean_identity_value(existing_applicant.name)
        same_identity = (
            (email and existing_email and email == existing_email)
            or (phone and existing_phone and phone == existing_phone)
            or (name and existing_name and name == existing_name)
        )
        same_strong_identity = (
            (email and existing_email and email == existing_email)
            or (phone and existing_phone and phone == existing_phone)
        )

        existing_exp_sig = _experience_signature(
            ExtractedExperience.query.filter_by(resume_id=existing_resume.id).all()
        )
        existing_edu_sig = _education_signature(
            ExtractedEducation.query.filter_by(resume_id=existing_resume.id).all()
        )

        same_structured_resume = (
            same_identity
            and (
                (exp_sig and existing_exp_sig and exp_sig == existing_exp_sig)
                or (
                    not exp_sig
                    and not existing_exp_sig
                    and edu_sig
                    and existing_edu_sig
                    and edu_sig == existing_edu_sig
                )
            )
        )
        same_exact_text = text_sig and text_sig == _resume_text_signature(existing_resume.original_text)

        if same_strong_identity or same_structured_resume or same_exact_text:
            return existing_resume

    return None

def _duplicate_resume_message(uploaded_filename, duplicate_resume):
    applicant = Applicant.query.get(duplicate_resume.applicant_id)
    job = JobDescription.query.get(duplicate_resume.job_id)
    candidate_name = applicant.name if applicant else 'Unknown Candidate'
    job_title = job.title if job else 'Unknown Job'

    return (
        f'{uploaded_filename}: duplicate resume blocked. Existing record: '
        f'{candidate_name} for {job_title} ({duplicate_resume.filename}).'
    )

def process_resume_file(file, job):
    filename = unique_upload_filename(file.filename)
    try:
        upload_timezone = ZoneInfo(current_app.config.get('DISPLAY_TIMEZONE', 'Asia/Manila'))
    except ZoneInfoNotFoundError:
        upload_timezone = timezone.utc
    upload_directory = job_upload_directory(
        current_app.config['UPLOAD_FOLDER'],
        job.id,
        job.title,
        datetime.now(upload_timezone),
    )
    os.makedirs(upload_directory, exist_ok=True)
    save_path = os.path.join(upload_directory, filename)
    file.save(save_path)

    try:
        duplicate_file = _find_duplicate_file(save_path)
        if duplicate_file:
            safe_delete_uploaded_file(
                save_path,
                current_app.config['UPLOAD_FOLDER'],
                legacy_upload_folder()
            )
            return False, _duplicate_resume_message(file.filename, duplicate_file)

        extracted_text = extract_text_from_file(save_path)
        extraction_status = 'success' if extracted_text else 'failed'

        if extraction_status != 'success':
            safe_delete_uploaded_file(
                save_path,
                current_app.config['UPLOAD_FOLDER'],
                legacy_upload_folder()
            )
            return False, f'{file.filename}: failed to extract text.'

        criteria = ScreeningCriteria.query.filter_by(job_id=job.id).first()
        min_score = criteria.min_fit_score if criteria else 50.0
        requires_all_critical = criteria.requires_all_critical if criteria else False

        required_skills_list = [s.strip() for s in job.required_skills.split(',') if s.strip()]
        critical_skills_list = [s.strip() for s in (job.critical_skills or '').split(',') if s.strip()]
        preferred_skills_list = [s.strip() for s in (job.preferred_skills or '').split(',') if s.strip()]

        evaluation = evaluate_candidate(
            resume_text=extracted_text,
            job_desc_text=f"{job.title} {job.required_skills} {job.critical_skills} {job.preferred_skills}",
            required_skills=required_skills_list,
            critical_skills=critical_skills_list,
            min_fit_score=min_score,
            experience_req=job.experience_req or 0,
            education_req=job.education_req,
            requires_all_critical=requires_all_critical,
            job_title=job.title,
            preferred_skills=preferred_skills_list
        )

        contact_info = evaluation['contact_info']
        duplicate_resume = _find_duplicate_resume(extracted_text, evaluation)
        if duplicate_resume:
            safe_delete_uploaded_file(
                save_path,
                current_app.config['UPLOAD_FOLDER'],
                legacy_upload_folder()
            )
            return False, _duplicate_resume_message(file.filename, duplicate_resume)

        applicant = Applicant(
            name=contact_info['name'],
            email=contact_info['email'],
            phone=contact_info['phone'],
            applied_job_id=job.id
        )
        db.session.add(applicant)
        db.session.flush()

        resume = Resume(
            applicant_id=applicant.id,
            job_id=job.id,
            uploaded_by=current_user.id,
            filename=filename,
            filepath=save_path,
            original_text=extracted_text,
            extraction_status=extraction_status
        )
        db.session.add(resume)
        db.session.flush()

        # Store the complete independently extracted skill inventory, not only
        # the subset that matched this job's configured requirements.
        for skill_name in evaluation['extracted_skills']:
            db.session.add(ExtractedSkill(resume_id=resume.id, skill_name=skill_name))

        for edu in evaluation['extracted_edu']:
            db.session.add(ExtractedEducation(
                resume_id=resume.id,
                degree=edu['degree'],
                institution=edu['institution']
            ))

        for exp in evaluation['extracted_exp']:
            db.session.add(ExtractedExperience(
                resume_id=resume.id,
                job_title=exp['job_title'],
                company=exp['company'],
                location=exp.get('location'),
                years=exp['years']
            ))

        for credential in evaluation['extracted_certifications']:
            db.session.add(ExtractedCertification(
                resume_id=resume.id,
                certification_name=credential['certification_name'],
                credential_type=credential['credential_type'],
                issuer=credential.get('issuer'),
                date_obtained=credential.get('date_obtained')
            ))

        result = ScreeningResult(
            resume_id=resume.id,
            applicant_id=applicant.id,
            job_id=job.id,
            skill_score=evaluation['skill_score'],
            experience_score=evaluation['experience_score'],
            education_score=evaluation['education_score'],
            text_similarity_score=evaluation['text_similarity_score'],
            fit_score=evaluation['fit_score'],
            recommendation_label=evaluation['recommendation_label'],
            confidence_level=evaluation['confidence_level'],
            confidence_reason=evaluation['confidence_reason'],
            summary=evaluation['summary'],
            decision_explanation=evaluation['decision_explanation']
        )
        result.set_matched_skills(evaluation['matched_skills'])
        result.set_missing_skills(evaluation['missing_skills'])
        result.set_matched_critical_skills(evaluation['matched_critical_skills'])
        result.set_missing_critical_skills(evaluation['missing_critical_skills'])
        db.session.add(result)
        db.session.flush()

        log = RecommendationLog(result_id=result.id, log_text=f"Auto-screened: {evaluation['summary']}")
        db.session.add(log)

        return True, f'{file.filename}: {evaluation["recommendation_label"]}'
    except Exception:
        safe_delete_uploaded_file(
            save_path,
            current_app.config['UPLOAD_FOLDER'],
            legacy_upload_folder()
        )
        raise

@resume_bp.route('/resume/upload', methods=['GET', 'POST'])
@login_required
@roles_required('hr', 'manager', 'admin')
def upload():
    jobs = JobDescription.query.all()
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        files = [file for file in request.files.getlist('file') if file.filename]
        job_id = request.form.get('job_id', type=int)
        
        if not files:
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if len(files) > MAX_RESUMES_PER_SCREENING:
            flash(f'You can upload up to {MAX_RESUMES_PER_SCREENING} resumes per screening.', 'danger')
            return redirect(request.url)

        if not job_id:
            flash('Target Job is required.', 'danger')
            return redirect(request.url)

        invalid_files = [file.filename for file in files if not allowed_file(file.filename)]
        if invalid_files:
            flash(f'Unsupported file type: {", ".join(invalid_files)}. Please upload PDF or DOCX resumes only.', 'danger')
            return redirect(request.url)

        job = JobDescription.query.get_or_404(job_id)
        processed = []
        failed = []

        for file in files:
            try:
                success, message = process_resume_file(file, job)
                if success:
                    processed.append(message)
                else:
                    failed.append(message)
            except Exception as exc:
                db.session.rollback()
                current_app.logger.error(
                    'Resume batch upload failed for %s\n%s',
                    file.filename,
                    traceback.format_exc()
                )
                failed.append(f'{file.filename}: {exc}')
            else:
                db.session.commit()

        if processed:
            flash(f'Processed {len(processed)} resume(s) successfully: {"; ".join(processed)}', 'success')
        if failed:
            flash(f'{len(failed)} resume(s) failed: {"; ".join(failed)}', 'danger')

        if processed:
            return redirect(url_for('screening.screening_results', job_id=job.id))
        else:
            return redirect(request.url)
                
    return render_template('resume/upload.html', jobs=jobs)
