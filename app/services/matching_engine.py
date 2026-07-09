from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_text_similarity(text1, text2):
    """
    Calculates the cosine similarity between two texts using TF-IDF.
    Uses n-gram range (1,2) to capture multi-word phrases like 'machine learning'.
    Returns a score between 0.0 and 100.0.
    """
    if not text1 or not text2:
        return 0.0
        
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity) * 100, 2)
    except ValueError:
        # Happens if texts are empty or contain no valid words
        return 0.0

def calculate_fit_score(skill_score, exp_score, edu_score, weights=(0.50, 0.30, 0.20)):
    """
    Calculates the final weighted fit score.
    Default weights: 50% skills, 30% experience, 20% education.
    """
    final_score = (skill_score * weights[0]) + (exp_score * weights[1]) + (edu_score * weights[2])
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
