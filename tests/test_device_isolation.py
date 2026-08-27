import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

from sqlalchemy import create_engine, inspect, text
from flask import g

from app import create_app, db
from app.models import (
    Applicant,
    ExtractedCertification,
    ExtractedEducation,
    ExtractedExperience,
    ExtractedSkill,
    JobDescription,
    RecommendationLog,
    Resume,
    ScreeningResult,
    User,
)
from app.routes.resume_routes import _find_duplicate_resume
from app.routes.summary_routes import build_summary
from app.utils.device import DEVICE_SESSION_KEY, LEGACY_DEVICE_ID
from migrations.device_isolation import DEVICE_OWNED_TABLES, migrate_device_isolation


class DeviceIsolationTestConfig:
    TESTING = True
    SECRET_KEY = 'device-isolation-test-key-with-32-characters'
    SQLALCHEMY_DATABASE_URI = 'sqlite://'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = tempfile.gettempdir()
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False


class DeviceIsolationTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(DeviceIsolationTestConfig)
        self.context = self.app.app_context()
        self.context.push()
        db.drop_all()
        db.create_all()

        user = User(username='shared-admin', password_hash='unused', role='admin')
        db.session.add(user)
        db.session.flush()
        job = JobDescription(
            title='Device Test Job',
            required_skills='Python',
            created_by=user.id,
        )
        db.session.add(job)
        db.session.commit()
        self.user_id = user.id
        self.job_id = job.id

        self.client_a = self.app.test_client()
        self.client_b = self.app.test_client()
        self.device_a = self._login_and_get_device(self.client_a)
        self.device_b = self._login_and_get_device(self.client_b)
        self.assertNotEqual(self.device_a, self.device_b)

        applicant = Applicant(
            device_id=self.device_a,
            name='Computer A Candidate',
            email='candidate@example.com',
            applied_job_id=self.job_id,
        )
        db.session.add(applicant)
        db.session.flush()
        resume = Resume(
            device_id=self.device_a,
            applicant_id=applicant.id,
            job_id=self.job_id,
            uploaded_by=self.user_id,
            filename='computer-a.pdf',
            filepath='computer-a.pdf',
            original_text='Computer A Candidate\nPython developer',
            extraction_status='success',
            upload_date=datetime(2026, 8, 27),
        )
        db.session.add(resume)
        db.session.flush()
        db.session.add_all([
            ExtractedSkill(
                device_id=self.device_a,
                resume_id=resume.id,
                skill_name='Python',
            ),
            ExtractedEducation(
                device_id=self.device_a,
                resume_id=resume.id,
                degree='BS Computer Science',
            ),
            ExtractedExperience(
                device_id=self.device_a,
                resume_id=resume.id,
                job_title='Developer',
                years=2,
            ),
            ExtractedCertification(
                device_id=self.device_a,
                resume_id=resume.id,
                certification_name='Python Certificate',
            ),
        ])
        result = ScreeningResult(
            device_id=self.device_a,
            resume_id=resume.id,
            applicant_id=applicant.id,
            job_id=self.job_id,
            skill_score=100,
            experience_score=100,
            education_score=100,
            fit_score=100,
            recommendation_label='Qualified',
            confidence_level='High',
            screened_at=datetime(2026, 8, 27),
        )
        result.set_matched_skills(['Python'])
        result.set_missing_skills([])
        db.session.add(result)
        db.session.flush()
        db.session.add(RecommendationLog(
            device_id=self.device_a,
            result_id=result.id,
            log_text='Auto-screened',
        ))
        db.session.commit()

        self.applicant_id = applicant.id
        self.resume_id = resume.id
        self.result_id = result.id

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.context.pop()

    def _login_and_get_device(self, client):
        client.get('/login')
        # The class-level app context used by this unittest keeps ``g`` alive
        # between test-client requests; real requests get a fresh app context.
        g.pop('_login_user', None)
        with client.session_transaction() as browser_session:
            device_id = browser_session[DEVICE_SESSION_KEY]
            browser_session['_user_id'] = str(self.user_id)
            browser_session['_fresh'] = True
        return device_id

    def test_same_account_only_sees_results_from_its_browser(self):
        page_a = self.client_a.get('/screening_results')
        page_b = self.client_b.get('/screening_results')

        self.assertIn(b'Computer A Candidate', page_a.data)
        self.assertNotIn(b'Computer A Candidate', page_b.data)
        self.assertIn(b'Showing 0 candidates.', page_b.data)

    def test_direct_detail_review_and_delete_urls_are_device_scoped(self):
        self.assertEqual(
            self.client_b.get(f'/screening_results/{self.result_id}').status_code,
            404,
        )
        self.assertEqual(
            self.client_b.post(
                f'/screening_results/{self.result_id}/review',
                data={'reviewer_status': 'Rejected', 'reviewer_notes': 'Denied'},
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client_b.post(
                f'/screening_results/{self.result_id}/delete'
            ).status_code,
            404,
        )

        result = db.session.get(ScreeningResult, self.result_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.reviewer_status, 'Pending Review')

    def test_bulk_delete_cannot_remove_another_device_records(self):
        response = self.client_b.post('/screening_results/delete_all')
        self.assertEqual(response.status_code, 302)
        self.assertIsNotNone(db.session.get(ScreeningResult, self.result_id))
        self.assertIsNotNone(db.session.get(Resume, self.resume_id))
        self.assertIsNotNone(db.session.get(Applicant, self.applicant_id))

    def test_shared_job_delete_cannot_remove_another_device_records(self):
        response = self.client_b.post(f'/jobs/{self.job_id}/delete')
        self.assertEqual(response.status_code, 302)
        self.assertIsNotNone(db.session.get(JobDescription, self.job_id))
        self.assertIsNotNone(db.session.get(ScreeningResult, self.result_id))
        self.assertIsNotNone(db.session.get(Resume, self.resume_id))

    def test_dashboard_and_executive_summary_counts_are_device_scoped(self):
        with patch(
            'app.routes.dashboard_routes.render_template',
            return_value='dashboard',
        ) as render_a:
            self.client_a.get('/dashboard')
        with patch(
            'app.routes.dashboard_routes.render_template',
            return_value='dashboard',
        ) as render_b:
            self.client_b.get('/dashboard')

        stats_a = render_a.call_args.kwargs['stats']
        stats_b = render_b.call_args.kwargs['stats']
        self.assertEqual(stats_a['total_applicants'], 1)
        self.assertEqual(stats_a['total_resumes'], 1)
        self.assertEqual(stats_a['qualified'], 1)
        self.assertEqual(stats_b['total_applicants'], 0)
        self.assertEqual(stats_b['total_resumes'], 0)
        self.assertEqual(stats_b['qualified'], 0)

        self.assertEqual(build_summary(device_id=self.device_a)['totals']['total_screened'], 1)
        self.assertEqual(build_summary(device_id=self.device_b)['totals']['total_screened'], 0)

    def test_duplicate_detection_is_device_specific(self):
        evaluation = {
            'contact_info': {
                'name': 'Computer A Candidate',
                'email': 'candidate@example.com',
                'phone': '',
            },
            'extracted_edu': [],
            'extracted_exp': [],
        }

        self.assertIsNotNone(_find_duplicate_resume(
            'Computer A Candidate\nPython developer',
            evaluation,
            device_id=self.device_a,
        ))
        self.assertIsNone(_find_duplicate_resume(
            'Computer A Candidate\nPython developer',
            evaluation,
            device_id=self.device_b,
        ))

    def test_device_id_persists_after_logout(self):
        self.client_a.post('/logout')
        self.client_a.get('/login')
        with self.client_a.session_transaction() as browser_session:
            self.assertEqual(browser_session[DEVICE_SESSION_KEY], self.device_a)
            self.assertTrue(browser_session.permanent)

    def test_legacy_static_upload_urls_are_not_public(self):
        response = self.client_a.get('/static/uploads/private-resume.pdf')
        self.assertEqual(response.status_code, 404)


class DeviceIsolationMigrationTests(unittest.TestCase):
    def test_migration_adds_indexes_and_quarantines_legacy_rows(self):
        engine = create_engine('sqlite://')
        with engine.begin() as connection:
            for table in DEVICE_OWNED_TABLES:
                connection.execute(text(
                    f'CREATE TABLE {table} (id INTEGER PRIMARY KEY)'
                ))
                connection.execute(text(f'INSERT INTO {table} (id) VALUES (1)'))

        first_run = migrate_device_isolation(engine)
        second_run = migrate_device_isolation(engine)

        self.assertTrue(first_run)
        self.assertEqual(second_run, [])
        inspector = inspect(engine)
        with engine.connect() as connection:
            for table in DEVICE_OWNED_TABLES:
                columns = {column['name'] for column in inspector.get_columns(table)}
                indexes = {index['name'] for index in inspector.get_indexes(table)}
                device_id = connection.execute(
                    text(f'SELECT device_id FROM {table} WHERE id = 1')
                ).scalar_one()
                self.assertIn('device_id', columns)
                self.assertIn(f'ix_{table}_device_id', indexes)
                self.assertEqual(device_id, LEGACY_DEVICE_ID)


if __name__ == '__main__':
    unittest.main()
