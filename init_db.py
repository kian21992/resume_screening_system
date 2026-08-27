import os
from app import create_app, db, bcrypt
from app.models import User


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
    db.session.commit()

    if created_usernames:
        print(f"Created default user(s): {', '.join(created_usernames)}")
    else:
        print("Default users already exist, skipping account creation.")

    # Jobs are browser/device-owned and therefore cannot be assigned safely by
    # an offline initializer that has no browser session. Create demo jobs from
    # the web interface on the browser that should own them.
    if not database_had_users:
        print("Create device-owned demo jobs from the web interface.")
