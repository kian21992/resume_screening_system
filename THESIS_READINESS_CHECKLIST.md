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
   - work experience score: 25% weight
   - education score: 15% weight
   - resume-to-job text similarity: 10% weight
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
- Skill matching uses word-boundary matching, then alias, fuzzy and stemmed fallbacks.
- Negated and aspirational wording is excluded, so "no experience with SQL" and a
  "currently learning" section do not count as evidence of a skill.
- Common technical aliases are supported, such as JS/JavaScript, NLP/Natural Language
  Processing, and SQL for MySQL/PostgreSQL/MSSQL.
- A skill can also be credited when the resume shows a technology that entails it
  (REST API from Django or API Gateway); the inferring technology is recorded so a
  reviewer can see the match was inferred rather than read.
- Raw resume-to-job cosine is rescaled onto 0-100 before it is weighted, so the
  10% component cannot cap the achievable fit score.
- Uploaded files are renamed uniquely to prevent overwriting another candidate's resume.
- New uploaded resumes are stored outside the public static folder.
- Candidate deletion only removes files from approved upload folders.

## Before Final Presentation

- Recreate the virtual environment on the actual/main project folder.
- Prepare a small labeled test set of resumes and expected outcomes.
- Take screenshots of the dashboard, upload page, ranking page, and candidate analysis page.
- Set a real `SECRET_KEY` environment variable before any hosted deployment.
- Use a clean database seeded with only demo jobs and demo resumes.
