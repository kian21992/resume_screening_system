import os
from app import create_app, db, bcrypt
from app.models import User, JobDescription, ScreeningCriteria


app = create_app()

with app.app_context():
    print("Creating tables...")
    db.create_all()

    # Ensure every local installation has the documented demonstration users.
    # Existing accounts keep their passwords and roles; only missing accounts
    # are created. This makes the initializer safe to rerun after pulling an
    # update without overwriting local account changes.
    database_had_users = User.query.first() is not None
    default_accounts = (
        ('hr_admin', 'password123', 'hr'),
        ('it_manager', 'password123', 'manager'),
        ('system_admin', 'password123', 'admin'),
    )
    default_users = {}
    created_usernames = []
    for username, password, role in default_accounts:
        user = User.query.filter_by(username=username).first()
        if user is None:
            user = User(
                username=username,
                password_hash=bcrypt.generate_password_hash(password).decode('utf-8'),
                role=role,
            )
            db.session.add(user)
            created_usernames.append(username)
        default_users[username] = user
    db.session.commit()

    if created_usernames:
        print(f"Created default user(s): {', '.join(created_usernames)}")
    else:
        print("Default users already exist, skipping account creation.")

    # Preserve the original behavior: sample jobs are populated only for a
    # fresh database, while missing default accounts can still be added to an
    # existing installation.
    if not database_had_users:
        print("Populating dummy data...")
        hr_user = default_users['hr_admin']

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
