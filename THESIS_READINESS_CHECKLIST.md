# Thesis Readiness Checklist

## Demo Flow

1. Log in to the system.
2. Create a job listing with required skills, preferred skills, education, experience, and screening criteria.
3. Upload at least three resumes for the same job:
   - one strong match
   - one partial match
   - one weak match
4. Open the Candidates page and confirm candidates are ranked by fit score.
5. Open a candidate detail page and explain:
   - required skill match score: 50% weight
   - work experience score: 30% weight
   - education score: 20% weight
   - text similarity score: reference metric only
6. Delete one candidate and confirm the record is removed.

## Verification Commands

Run these from the project folder after recreating the venv:

```cmd
python -m unittest discover -s tests
python -m compileall app tests
python app.py
```

## Minimum Defense Points

- The system uses deterministic scoring so recommendations are explainable.
- Skill matching uses exact word-boundary matching to reduce false positives.
- Common technical aliases are supported, such as JS/JavaScript and NLP/Natural Language Processing.
- Uploaded files are renamed uniquely to prevent overwriting another candidate's resume.
- New uploaded resumes are stored outside the public static folder.
- Candidate deletion only removes files from approved upload folders.

## Before Final Presentation

- Recreate the virtual environment on the actual/main project folder.
- Prepare a small labeled test set of resumes and expected outcomes.
- Take screenshots of the dashboard, upload page, ranking page, and candidate analysis page.
- Set a real `SECRET_KEY` environment variable before any hosted deployment.
- Use a clean database seeded with only demo jobs and demo resumes.
