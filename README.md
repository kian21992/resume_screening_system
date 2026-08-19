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

## Notes

Local files such as `.env`, `venv/`, uploaded resumes, cache files, and the SQLite database are intentionally not uploaded to GitHub. They are created on each computer during setup and use.

New resumes are stored outside the public static folder and organized by job and
upload date:

```text
instance/uploads/jobs/<job-id>-<job-title>/<year>/<month>/week-<number>/<unique-filename>
```

Weeks are grouped by day of the month: 1-7, 8-14, 15-21, 22-28, and 29-31.
Existing resumes keep their original stored paths.
