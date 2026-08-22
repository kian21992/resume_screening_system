from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Single source of truth for the fit-score weights, matching the published
# model: skills 50%, experience 25%, education 15%, text similarity 10%.
# The UI, the generated narrative, and the score breakdown all read these
# rather than restating the numbers, so the explanation shown to a reviewer
# always reconstructs the score.
FIT_WEIGHTS = (0.50, 0.25, 0.15, 0.10)
FIT_WEIGHT_PERCENTS = tuple(int(round(w * 100)) for w in FIT_WEIGHTS)


def calculate_text_similarity(text1, text2, corpus=None):
    """
    Cosine similarity between two texts over TF-IDF vectors.

    ``corpus`` is the document collection the IDF term is computed from. Pass
    the other resumes being screened; inverse *document* frequency only carries
    information when it is fitted across many documents. Without it the two
    input texts are the entire corpus, IDF collapses to two possible values,
    and the result is cosine over term frequencies wearing a TF-IDF label.

    Reported as a diagnostic only — see ``calculate_fit_score``.
    Returns a score between 0.0 and 100.0.
    """
    if not text1 or not text2:
        return 0.0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    try:
        fit_documents = [text1, text2]
        if corpus:
            fit_documents = [text1, text2, *[doc for doc in corpus if doc]]
        vectorizer.fit(fit_documents)
        matrix = vectorizer.transform([text1, text2])
        similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(float(similarity) * 100, 2)
    except ValueError:
        # Happens if texts are empty or contain no valid words
        return 0.0


def calculate_fit_score(skill_score, exp_score, edu_score,
                        text_similarity_score=None, weights=None):
    """
    Calculates the final weighted fit score: 50% skills, 25% experience,
    15% education, 10% resume-to-job text similarity.

    Omitting ``text_similarity_score`` falls back to the three-component model
    (50/30/20) so the component weights still sum to 1.0 rather than silently
    scoring every candidate out of 90.
    """
    if text_similarity_score is None:
        w = weights or (0.50, 0.30, 0.20)
        final_score = (skill_score * w[0]) + (exp_score * w[1]) + (edu_score * w[2])
    else:
        w = weights or FIT_WEIGHTS
        final_score = ((skill_score * w[0]) + (exp_score * w[1]) +
                       (edu_score * w[2]) + (text_similarity_score * w[3]))
    return round(min(final_score, 100.0), 2)


def calculate_skills_match(resume_skills, required_skills):
    """
    Calculates the percentage of required skills found in the resume.
    """
    if not required_skills:
        return 100.0  # If no skills required, it's a match

    resume_skills_lower = [s.lower() for s in resume_skills]
    matched_skills = [s for s in required_skills if s.lower() in resume_skills_lower]

    match_percentage = (len(matched_skills) / len(required_skills)) * 100
    return round(match_percentage, 2)
