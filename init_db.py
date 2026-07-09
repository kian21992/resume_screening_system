import os
from app import create_app, db, bcrypt
from app.models import User, JobDescription, ScreeningCriteria


app = create_app()

with app.app_context():
    print("Creating tables...")
    db.create_all()

    # Check if we already have users
    if not User.query.first():
        print("Populating dummy data...")
        
        # Create users
        hr_user = User(username='hr_admin', password_hash=bcrypt.generate_password_hash('password123').decode('utf-8'), role='hr')
        manager_user = User(username='it_manager', password_hash=bcrypt.generate_password_hash('password123').decode('utf-8'), role='manager')
        db.session.add_all([hr_user, manager_user])
        db.session.commit()

        # Create dummy jobs
        job1 = JobDescription(
            title='Software Engineer',
            required_skills='Python, Django, SQL, REST API',
            critical_skills='Python, SQL',
            preferred_skills='Docker, AWS, React',
            education_req='Bachelor of Science in Computer Science',
            experience_req=3,
            created_by=hr_user.id
        )
        
        job2 = JobDescription(
            title='Data Scientist',
            required_skills='Python, Machine Learning, Pandas, Scikit-learn',
            critical_skills='Python, Machine Learning',
            preferred_skills='Deep Learning, NLP, TensorFlow',
            education_req='Master in Data Science or related',
            experience_req=2,
            created_by=hr_user.id
        )
        
        db.session.add_all([job1, job2])
        db.session.commit()
        
        # Create screening criteria for jobs
        criteria1 = ScreeningCriteria(job_id=job1.id, min_fit_score=60.0, requires_all_critical=True)
        criteria2 = ScreeningCriteria(job_id=job2.id, min_fit_score=70.0, requires_all_critical=False)
        
        db.session.add_all([criteria1, criteria2])
        db.session.commit()
        
        print("Dummy data populated successfully.")
    else:
        print("Database already contains data, skipping dummy data population.")
