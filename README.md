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

## Run with Docker (local thesis demo)

### Run the published image (Windows PowerShell)

Install and start Docker Desktop, then run these commands in PowerShell. This
method pulls the prebuilt image from Docker Hub, so it does not require a local
Python installation or a clone of this repository.

1. Check Docker and confirm its engine is running:

   ~~~powershell
   docker --version
   docker info
   ~~~

   If docker info cannot connect, start Docker Desktop and wait for it to finish
   starting.

2. Create a folder for this installation:

   ~~~powershell
   cd $HOME\Desktop
   mkdir ARSS-Thesis
   cd ARSS-Thesis
   ~~~

   Create the folder only once. On later runs, return to the same ARSS-Thesis
   folder so you keep using its .env file.

3. Download the image and check that it is present:

   ~~~powershell
   docker pull arssthesis2026/resume-screening-system:1.0.0
   docker images
   ~~~

4. Generate a private secret key and create .env in the current folder:

   ~~~powershell
   $arssSecret = docker run --rm --entrypoint python arssthesis2026/resume-screening-system:1.0.0 -c "import secrets; print(secrets.token_urlsafe(48))"
   "SECRET_KEY=$arssSecret" | Set-Content .env
   Get-ChildItem -Force
   notepad .env
   ~~~

   Notepad opens .env. Keep the generated SECRET_KEY line and add these account
   settings. Replace each password placeholder with a different private
   password containing at least 12 characters:

   ~~~dotenv
   SECRET_KEY=your-existing-generated-secret-key
   INITIAL_ADMIN_USERNAME=system_admin
   INITIAL_ADMIN_PASSWORD=your-private-admin-password
   INITIAL_MANAGER_USERNAME=it_manager
   INITIAL_MANAGER_PASSWORD=your-private-manager-password
   INITIAL_HR_USERNAME=hr_admin
   INITIAL_HR_PASSWORD=your-private-hr-password
   ~~~

   The SECRET_KEY line above shows where the already generated value belongs.
   Do not replace your generated key with the literal example text. Save and
   close Notepad. Keep .env private and reuse the same key on later runs; browser
   identities are signed with it.

5. Start the application and check that the container is running:

   ~~~powershell
   docker run -d --name arss-thesis -p 127.0.0.1:5000:5000 --env-file .env -e APP_ENV=development -e DATABASE_URL=sqlite:////app/instance/resume_screening.db -e UPLOAD_FOLDER=/app/instance/uploads -v arss_thesis_data:/app/instance arssthesis2026/resume-screening-system:1.0.0
   docker ps
   ~~~

   If arss-thesis does not appear in docker ps, inspect its output with
   docker logs arss-thesis.

6. Create the admin, manager, and HR users from the private passwords in .env:

   ~~~powershell
   docker exec arss-thesis python -m flask --app app bootstrap-users
   ~~~

   This command creates missing users. It does not change passwords for users
   already in the database.

7. Open <http://localhost:5000> and sign in with an account from .env.

The arss_thesis_data Docker volume stores the SQLite database and uploaded
resumes. Use docker stop arss-thesis and docker start arss-thesis to stop and
restart the same container. Keep .env in the ARSS-Thesis folder; changing that
file alone does not update an already created container. The sample passwords
in the Default Accounts section below apply to the init_db.py setup, not to
the private accounts created with bootstrap-users.

### Build from this repository with Docker Compose

Install Docker Desktop (or Docker Engine with the Compose plugin), then run these
commands from the repository folder. On Windows PowerShell, create your private
configuration file with:

```powershell
Copy-Item .env.example .env
docker run --rm python:3.11-slim-bookworm python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Copy the generated value into `SECRET_KEY=` in `.env`. Keep the same value on
later runs: browser identities are signed with this key. Never commit `.env`.
On macOS or Linux, use `cp .env.example .env` instead of `Copy-Item`.

Build the image, create the local demonstration accounts, and start the app:

```bash
docker compose build
docker compose run --rm web python init_db.py
docker compose up -d
```

Open <http://127.0.0.1:5000> and sign in with one of the demonstration accounts
listed below. To see startup output or stop the app:

```bash
docker compose logs -f web
docker compose down
```

The `app_data` Docker volume stores the SQLite database and uploaded resumes,
so ordinary `docker compose down` and later restarts preserve them. Running
`docker compose down --volumes` deletes that demo data. The image includes the
spaCy model and NLTK data; no separate NLP installation is needed. After code or
dependency changes, rebuild with `docker compose up --build -d`.

This Compose setup serves the local demo over HTTP and binds only to your own
computer. It uses the documented sample passwords, so do not expose it publicly.
The existing Render deployment uses its own PostgreSQL service and deployment
settings rather than this local Compose setup.

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
| Create or edit jobs | Yes | Yes | Yes |
| Delete jobs or candidates | Yes | No | Yes |

Each upload targets one posted job. The same applicant can be screened for a
different job in a separate upload, with a separate score and recommendation.
Duplicate uploads are blocked only within the same job and browser.

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
device-owned candidate data. Historical candidate rows belonging to a different
device are preserved on a hidden legacy job copy so they cannot block safe job
management or leak job details. Records without trustworthy browser ownership
are assigned a reserved legacy identifier that no browser can receive,
quarantining them from the user interface. Fresh databases created after this
change already contain the required columns and indexes; the migration command
will report that the schema is up to date.

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
