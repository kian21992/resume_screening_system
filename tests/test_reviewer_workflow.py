import tempfile
import unittest
from datetime import datetime

from app import create_app, db
from app.models import (
    Applicant,
    ExtractedEducation,
    ExtractedExperience,
    JobDescription,
    RecommendationLog,
    Resume,
    ScreeningResult,
    User,
)
from app.routes.resume_routes import _duplicate_resume_message, _find_duplicate_resume


class ReviewerTestConfig:
    TESTING = True
    SECRET_KEY = "reviewer-test-key"
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = tempfile.gettempdir()


class TestReviewerWorkflow(unittest.TestCase):

    def setUp(self):
        self.app = create_app(ReviewerTestConfig)
        self.context = self.app.app_context()
        self.context.push()

        user = User(username="reviewer", password_hash="test", role="hr")
        db.session.add(user)
        db.session.flush()
        job = JobDescription(
            title="Teacher",
            required_skills="Classroom Management",
            critical_skills="",
            preferred_skills="",
            education_req="Bachelor's",
            experience_req=2,
            created_by=user.id,
        )
        db.session.add(job)
        db.session.flush()
        applicant = Applicant(
            name="Maria Santos",
            email="maria@example.com",
            phone="09171234567",
            applied_job_id=job.id,
        )
        db.session.add(applicant)
        db.session.flush()
        resume = Resume(
            applicant_id=applicant.id,
            job_id=job.id,
            uploaded_by=user.id,
            filename="maria.docx",
            filepath="maria.docx",
            original_text=(
                "Maria Santos\nSkills\nClassroom Management\n"
                "Professional Experience\nTeacher at Sample School\n"
                "January 2020 - January 2024\nEducation\nBachelor of Education"
            ),
            extraction_status="success",
        )
        db.session.add(resume)
        db.session.flush()
        result = ScreeningResult(
            resume_id=resume.id,
            applicant_id=applicant.id,
            job_id=job.id,
            screened_at=datetime(2026, 6, 23, 0, 30),
            fit_score=82,
            skill_score=100,
            experience_score=100,
            education_score=100,
            recommendation_label="Qualified",
            confidence_level="High",
            confidence_reason="Confidence support: resume text was readable enough for screening.",
            summary="The candidate meets the configured requirements.",
        )
        result.set_matched_skills(["Classroom Management"])
        result.set_missing_skills([])
        db.session.add(result)
        db.session.flush()

        second_applicant = Applicant(
            name="Ana Reyes",
            email="ana@example.com",
            phone="09181234567",
            applied_job_id=job.id,
        )
        db.session.add(second_applicant)
        db.session.flush()
        second_resume = Resume(
            applicant_id=second_applicant.id,
            job_id=job.id,
            filename="ana.docx",
            filepath="ana.docx",
            original_text="Ana Reyes\nTeaching background\n",
            extraction_status="success",
        )
        db.session.add(second_resume)
        db.session.flush()
        second_result = ScreeningResult(
            resume_id=second_resume.id,
            applicant_id=second_applicant.id,
            job_id=job.id,
            fit_score=45,
            skill_score=0,
            experience_score=50,
            education_score=100,
            recommendation_label="Not Qualified",
            reviewer_status="Rejected",
            confidence_level="High",
            confidence_reason="Clear required-skill gap.",
            summary="The candidate does not meet the configured requirements.",
        )
        second_result.set_matched_skills([])
        second_result.set_missing_skills(["Classroom Management"])
        db.session.add(second_result)
        db.session.commit()

        self.user_id = user.id
        self.job_id = job.id
        self.result_id = result.id
        self.second_result_id = second_result.id
        self.client = self.app.test_client()
        with self.client.session_transaction() as session:
            session["_user_id"] = str(self.user_id)
            session["_fresh"] = True

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.context.pop()

    def test_reviewer_can_save_decision_notes_and_attribution(self):
        response = self.client.post(
            f"/screening_results/{self.result_id}/review",
            data={
                "reviewer_status": "Needs Clarification",
                "reviewer_notes": (
                    "Resume lacks teaching license proof, but has strong classroom experience."
                ),
            },
        )
        self.assertEqual(response.status_code, 302)

        result = db.session.get(ScreeningResult, self.result_id)
        self.assertEqual(result.reviewer_status, "Needs Clarification")
        self.assertIn("teaching license proof", result.reviewer_notes)
        self.assertEqual(result.reviewed_by, self.user_id)
        self.assertIsNotNone(result.reviewed_at)
        self.assertEqual(result.recommendation_label, "Qualified")
        self.assertEqual(RecommendationLog.query.filter_by(result_id=self.result_id).count(), 1)

    def test_invalid_status_is_rejected(self):
        self.client.post(
            f"/screening_results/{self.result_id}/review",
            data={"reviewer_status": "Automatically Hired", "reviewer_notes": "Invalid"},
        )
        result = db.session.get(ScreeningResult, self.result_id)
        self.assertEqual(result.reviewer_status, "Pending Review")
        self.assertIsNone(result.reviewed_at)

    def test_oversized_notes_are_rejected(self):
        self.client.post(
            f"/screening_results/{self.result_id}/review",
            data={"reviewer_status": "Reviewed", "reviewer_notes": "x" * 3001},
        )
        result = db.session.get(ScreeningResult, self.result_id)
        self.assertEqual(result.reviewer_status, "Pending Review")
        self.assertIsNone(result.reviewer_notes)

    def test_detail_and_results_pages_show_human_review_controls(self):
        detail = self.client.get(f"/screening_results/{self.result_id}")
        listing = self.client.get("/screening_results")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(b"Human Review", detail.data)
        self.assertIn(b"Audit Trail", detail.data)
        self.assertIn(b"Uploaded By", detail.data)
        self.assertIn(b"Date Screened", detail.data)
        self.assertIn(b"Date Reviewed", detail.data)
        self.assertIn(b"Jun 23, 2026 08:30 AM", detail.data)
        self.assertIn(b"Time zone: Asia/Manila", detail.data)
        self.assertIn(b"reviewer", detail.data)
        self.assertIn(b"Reviewer Notes", detail.data)
        self.assertIn(b"Move to Interview", detail.data)
        self.assertIn(b"Decision Confidence", detail.data)
        self.assertIn(b"High Confidence", detail.data)
        self.assertEqual(listing.status_code, 200)
        self.assertIn(b"Pending Review", listing.data)
        self.assertIn(b"High Confidence", listing.data)

    def test_results_page_filters_by_reviewer_decision(self):
        response = self.client.get("/screening_results?reviewer_status=Rejected")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Ana Reyes", response.data)
        self.assertNotIn(b"Maria Santos", response.data)
        self.assertIn(b"All Reviewer Decisions", response.data)
        self.assertIn(b"Rejected", response.data)

    def test_duplicate_resume_is_detected_from_identity_and_work_history(self):
        resume = Resume.query.filter_by(filename="maria.docx").first()
        db.session.add(ExtractedEducation(
            resume_id=resume.id,
            degree="Bachelor of Education",
            institution="State University",
        ))
        db.session.add(ExtractedExperience(
            resume_id=resume.id,
            job_title="Teacher",
            company="Sample School",
            location="Manila",
            years=4,
        ))
        db.session.commit()

        evaluation = {
            "contact_info": {
                "name": "Maria Santos",
                "email": "maria@example.com",
                "phone": "09171234567",
            },
            "extracted_edu": [
                {"degree": "Bachelor of Education", "institution": "State University"},
            ],
            "extracted_exp": [
                {
                    "job_title": "Teacher",
                    "company": "Sample School",
                    "location": "Manila",
                    "years": 4,
                },
            ],
        }

        duplicate = _find_duplicate_resume(
            "Maria Santos\nTeacher\nSample School\nJanuary 2020 - January 2024",
            evaluation,
        )

        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate.id, resume.id)

    def test_duplicate_resume_is_detected_from_equivalent_text(self):
        evaluation = {
            "contact_info": {
                "name": "Unknown Candidate",
                "email": "Unknown Email",
                "phone": "Unknown Phone",
            },
            "extracted_edu": [],
            "extracted_exp": [],
        }

        duplicate = _find_duplicate_resume(
            "Maria Santos Skills Classroom Management Professional Experience Teacher at Sample School January 2020 January 2024 Education Bachelor of Education",
            evaluation,
        )

        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate.filename, "maria.docx")

    def test_duplicate_resume_is_detected_from_email_even_when_structure_differs(self):
        evaluation = {
            "contact_info": {
                "name": "Maria Santos",
                "email": "maria@example.com",
                "phone": "Unknown Phone",
            },
            "extracted_edu": [],
            "extracted_exp": [],
        }

        duplicate = _find_duplicate_resume(
            "Different PDF text layout, but the same candidate email appears.",
            evaluation,
        )

        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate.filename, "maria.docx")

    def test_duplicate_warning_names_existing_candidate_and_job(self):
        resume = Resume.query.filter_by(filename="maria.docx").first()

        message = _duplicate_resume_message("copy.pdf", resume)

        self.assertIn("duplicate resume blocked", message)
        self.assertIn("Maria Santos", message)
        self.assertIn("Teacher", message)
        self.assertIn("maria.docx", message)

    def test_results_page_combines_job_and_reviewer_filters(self):
        response = self.client.get(
            f"/screening_results?job_id={self.job_id}&reviewer_status=Pending+Review"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Maria Santos", response.data)
        self.assertNotIn(b"Ana Reyes", response.data)

    def test_results_page_sorts_by_lowest_fit_score(self):
        response = self.client.get("/screening_results?sort=fit_asc")
        self.assertEqual(response.status_code, 200)
        self.assertLess(response.data.find(b"Ana Reyes"), response.data.find(b"Maria Santos"))

    def test_results_page_sorts_by_actual_screening_date(self):
        newest = self.client.get("/screening_results?sort=newest")
        oldest = self.client.get("/screening_results?sort=oldest")
        self.assertLess(newest.data.find(b"Ana Reyes"), newest.data.find(b"Maria Santos"))
        self.assertLess(oldest.data.find(b"Maria Santos"), oldest.data.find(b"Ana Reyes"))

    def test_sort_selection_applies_immediately(self):
        response = self.client.get("/screening_results")
        self.assertIn(b'name="sort" onchange="this.form.submit()"', response.data)


if __name__ == "__main__":
    unittest.main()
