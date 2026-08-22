# Resume Screening System

A Flask-based resume screening system for uploading resumes, extracting candidate details, matching skills against job descriptions, and reviewing screening results.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/kian21992/resume_screening_system.git
cd resume_screening_system
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Install the NLP data used by the system:

```bash
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
python -m nltk.downloader punkt stopwords
```

5. Create a local environment file:

```bash
copy .env.example .env
```

Replace `SECRET_KEY` with a random value of at least 32 characters. You can
generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

6. Initialize the database with sample users and jobs:

```bash
python init_db.py
```

7. Run the application:

```bash
python app.py
```

Open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## Default Accounts

The database initializer creates these sample accounts:

```text
Username: hr_admin
Password: password123
Role: HR
```

```text
Username: it_manager
Password: password123
Role: Manager
```

These are local demonstration credentials. Change them before deploying the
application and never use the sample passwords on a public server.

## Roles and permissions

| Action | HR | Manager | Admin |
| --- | --- | --- | --- |
| View jobs and candidates | Yes | Yes | Yes |
| Upload resumes and review candidates | Yes | Yes | Yes |
| Create or edit jobs | No | Yes | Yes |
| Delete jobs or candidates | No | No | Yes |

The initializer does not create an admin account. Assign the admin role to an
existing trusted user with:

```bash
python -m flask --app app set-user-role it_manager admin
```

Role restrictions are enforced by the server; the matching controls are also
hidden from users who do not have permission.

## Deployment security

Set `APP_ENV=production` when deploying. In production the application refuses
to start unless `SECRET_KEY` contains at least 32 characters, and session
cookies are marked secure for HTTPS. Login attempts are limited to five per
minute per client IP, and every state-changing form requires a CSRF token.

The default `memory://` rate-limit storage is intended for local or single-process
use. Configure `RATELIMIT_STORAGE_URI` with a shared backend such as Redis when
the deployment runs multiple workers. If the application is behind a reverse
proxy, configure trusted proxy handling at the hosting layer so the application
receives the correct client address.

## Notes

Local files such as `.env`, `venv/`, uploaded resumes, cache files, and the SQLite database are intentionally not uploaded to GitHub. They are created on each computer during setup and use.

New resumes are stored outside the public static folder and organized by job and
upload date:

```text
instance/uploads/jobs/<job-id>-<job-title>/<year>/<month>/week-<number>/<unique-filename>
```

Weeks are grouped by day of the month: 1-7, 8-14, 15-21, 22-28, and 29-31.
Existing resumes keep their original stored paths.
