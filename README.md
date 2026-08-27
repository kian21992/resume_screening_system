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

6. Initialize the database with sample users:

```bash
python init_db.py
```

Jobs are device-owned, so create demonstration jobs from the web interface in
the browser profile that should own them.

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

```text
Username: system_admin
Password: password123
Role: Admin
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

The initializer creates `system_admin` for local demonstrations. You can also
assign the admin role to another existing trusted user with:

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

## Device-isolated screening data

Each browser profile receives a random 256-bit `device_id` in Flask's signed,
persistent session cookie. Applicants, resumes, extracted fields, screening
results, recommendations, rankings, duplicate checks, dashboard totals, and
executive-summary totals are scoped to that identifier on the server. Jobs and
their screening criteria are device-owned as well. Job lists, upload choices,
and direct job URLs are filtered by the same identifier.

The identifier survives login/logout and normal browser restarts. Clearing site
cookies or using private browsing creates a new device identity, so the previous
browser's records will no longer be visible from that browser profile. Copying a
valid session cookie also copies its device identity; always use HTTPS and keep
`SECRET_KEY` private in production.

Before starting an upgraded installation against an existing database, back up
the database and run the idempotent migration:

```bash
python -m flask --app app migrate-device-isolation
```

For existing jobs, the migration infers ownership from their oldest
device-owned candidate data. Records without trustworthy browser ownership are
assigned a reserved legacy identifier that no browser can receive, quarantining
them from the user interface. Fresh databases created after this change already
contain the required columns and indexes; the migration command will report that
the schema is up to date.

Run the device-isolation and security regression tests with:

```bash
python -m pytest -q tests/test_device_isolation.py tests/test_security.py tests/test_reviewer_workflow.py
```

Run the complete test suite with:

```bash
python -m pytest -q
```

## Render thesis deployment

The repository includes a `render.yaml` Blueprint for a free, temporary thesis
deployment in Render's Singapore region. It provisions one Flask web service
and one private Render Postgres database. Render prompts for these secrets when
you create the Blueprint:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/kian21992/resume_screening_system)

- `INITIAL_ADMIN_USERNAME`
- `INITIAL_ADMIN_PASSWORD` (at least 12 characters)
- `INITIAL_MANAGER_PASSWORD` (at least 12 characters; creates `it_manager`)
- `INITIAL_HR_PASSWORD` (at least 12 characters; creates `hr_admin`)

The deployment generates `SECRET_KEY`, installs the pinned spaCy model and NLTK
data, runs the device-isolation migration, creates the initial administrator,
manager, and HR users only when they are missing, and starts the application
with Gunicorn. The manager and HR usernames can be overridden with
`INITIAL_MANAGER_USERNAME` and `INITIAL_HR_USERNAME` environment variables.

To deploy:

1. Commit and push the repository to GitHub.
2. In Render, choose **New > Blueprint** and connect this repository.
3. Keep the `main` branch and `render.yaml` path selected.
4. Enter private administrator, manager, and HR passwords when prompted.
5. Apply the Blueprint and wait for both the database and web service to become
   available.
6. Open the generated `onrender.com` URL and sign in with the administrator.

The free web service has an ephemeral filesystem. Uploaded resume files remain
on the server only until Render restarts or redeploys the instance, although
database-backed extracted text and screening results remain until the free
database expires. Use only synthetic resumes for this thesis demo. Upgrade the
web service and attach a persistent disk before storing real applicant files.

## Notes

Local files such as `.env`, `venv/`, uploaded resumes, cache files, and the SQLite database are intentionally not uploaded to GitHub. They are created on each computer during setup and use.

New resumes are stored outside the public static folder and organized by job and
upload date:

```text
instance/uploads/jobs/<job-id>-<job-title>/<year>/<month>/week-<number>/<unique-filename>
```

Weeks are grouped by day of the month: 1-7, 8-14, 15-21, 22-28, and 29-31.
Existing resumes keep their original stored paths.
