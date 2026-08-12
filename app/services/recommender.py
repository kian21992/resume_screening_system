def generate_recommendation(fit_score, min_fit_score=50.0):
    """
    Generates a recommendation label based on the fit score.
    Logic:
    - Qualified: >= 75%
    - For Review: >= min_fit_score (e.g., 50%) but < 75%
    - Not Qualified: < min_fit_score
    """
    if fit_score >= 75.0:
        return "Qualified"
    elif fit_score >= min_fit_score:
        return "For Review"
    else:
        return "Not Qualified"

import re as _re
import hashlib

SKILL_ALIASES = {
    "javascript": ["js", "ecmascript", "es6"],
    "js": ["javascript", "ecmascript", "es6"],
    "react": ["react.js", "reactjs"],
    "node": ["node.js", "nodejs"],
    "node.js": ["node", "nodejs"],
    "html": ["html5"],
    "css": ["css3"],
    "python": ["python3", "python 3"],
    "postgres": ["postgresql"],
    "postgresql": ["postgres", "postgre sql"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "natural language processing": ["nlp"],
    "lesson planning": ["lesson plan", "lesson plans", "develop lesson plans", "developlessonplans", "curriculum planning", "prepared and delivered lessons", "designs engaging lesson plans"],
    "classroom management": ["classroommanagement", "classroom management strategies", "managed classroom instruction"],
    "communication skills": ["communication skill", "written and verbal communication", "verbal and written communication skills"],
    "c#": ["c sharp"],
    "c sharp": ["c#"],
}


def _skill_variants(skill):
    normalized = skill.strip().lower()
    variants = {normalized}
    variants.update(SKILL_ALIASES.get(normalized, []))
    return variants


def _literal_skill_found(resume_text_lower, skill_variant):
    if _re.search(r'[^a-z0-9\s]', skill_variant):
        return skill_variant in resume_text_lower

    words = skill_variant.split()
    if len(words) > 1:
        pattern = r'\b' + r'\s+'.join(_re.escape(word) for word in words) + r'\b'
    else:
        pattern = r'\b' + _re.escape(skill_variant) + r'\b'

    return bool(_re.search(pattern, resume_text_lower))

def _pick_phrase(options, seed, offset=0):
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    index = (int(digest[:8], 16) + offset) % len(options)
    return options[index]

def analyze_skills(resume_skills_text, required_skills_list):
    """
    Identifies matched and missing skills using word-boundary regex so that
    short skills like 'R', 'C', 'Go' are not falsely matched inside longer words
    (e.g. 'R' should not match 'recruiter', 'Go' should not match 'algorithm').
    """
    matched = []
    missing = []
    resume_text_lower = resume_skills_text.lower()

    for req_skill in required_skills_list:
        skill = req_skill.strip()
        if not skill:
            continue
        found = any(_literal_skill_found(resume_text_lower, variant) for variant in _skill_variants(skill))

        if found:
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing


def analyze_preferred_skills(resume_skills_text, preferred_skills_list):
    """
    Checks how many preferred (nice-to-have) skills appear in the resume.
    Returns (matched_preferred, total_preferred, bonus_pct) where bonus_pct
    is an additive bonus capped at 10.0 percentage points applied to the
    overall skill score.
    """
    if not preferred_skills_list:
        return [], 0, 0.0

    resume_text_lower = resume_skills_text.lower()
    matched_preferred = []

    for pref_skill in preferred_skills_list:
        skill = pref_skill.strip()
        if not skill:
            continue
        found = any(_literal_skill_found(resume_text_lower, variant) for variant in _skill_variants(skill))
        if found:
            matched_preferred.append(skill)

    total_preferred = len([s for s in preferred_skills_list if s.strip()])
    if total_preferred == 0:
        return [], 0, 0.0

    # Bonus: up to 10 points scaled by how many preferred skills were found
    bonus_pct = round((len(matched_preferred) / total_preferred) * 10.0, 2)
    return matched_preferred, total_preferred, bonus_pct

def get_degree_rank(degree_str):
    import re as _re_rank
    if not degree_str:
        return 0
    degree_lower = degree_str.lower()

    # Use word-boundary patterns to avoid substring false positives
    # (e.g. "diploma" contains "ma", "ms", "ph" — plain substring checks misfire).
    def _matches(patterns):
        return any(_re_rank.search(p, degree_lower) for p in patterns)

    if _matches([r'\bph\.?d\b', r'\bdoctor']):
        return 5
    if _matches([
        r'\bmaster', r'\bm\.s\.?\b', r'\bm\.a\.?\b', r'\bmba\b',
        r'\b(?:med|maed|mpa|msw|mtech|msc)\b',
    ]):
        return 4
    if _matches([
        r'\bbachelor', r'\bb\.s\.?\b', r'\bb\.a\.?\b', r'\bb\.e\.?\b',
        r'\bbtech\b', r'\bbs\b', r'\bba\b', r'\bbe\b',
        r'\b(?:bsed|beed|bsn|bsit|bscs|bsa|bba|bshm|bsba|ab)\b',
    ]):
        return 3
    if _matches([r'\bassociate']):
        return 2
    if _matches([r'\bhigh\s+school', r'\bdiploma\b', r'\bged\b']):
        return 1
    return 0

def _degree_label(rank):
    """Returns a human-readable degree label from a rank integer."""
    labels = {5: "Ph.D.", 4: "Master's degree", 3: "Bachelor's degree",
              2: "Associate's degree", 1: "High School diploma", 0: "unverified education"}
    return labels.get(rank, "unverified education")

def generate_analysis_narrative(job_title, fit_score, skill_score, exp_score, edu_score,
                                matched_skills, missing_skills, total_exp_years,
                                extracted_edu, experience_req, education_req,
                                disqualified_by_critical_skills,
                                matched_preferred=None, preferred_bonus=0.0,
                                matched_critical_skills=None, missing_critical_skills=None,
                                disqualification_reason=None):
    """
    Generates a unique, candidate-specific analysis narrative using all
    available evaluation signals. Each section is tailored to the actual data.
    """
    total_required = len(matched_skills) + len(missing_skills)
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)
    matched_critical_skills = matched_critical_skills or []
    missing_critical_skills = missing_critical_skills or []

    # --- Determine fit tier label ---
    if fit_score >= 75:
        tier = "strong"
    elif fit_score >= 50:
        tier = "moderate"
    else:
        tier = "weak"

    # --- Opening sentence: overall verdict ---
    if disqualification_reason:
        opening = (
            f"This candidate is marked Not Qualified for the {job_title} role because "
            f"{disqualification_reason}. The overall fit score remains {fit_score:.0f}% "
            f"to reflect the weighted skills, experience, and education evidence separately from the rule outcome."
        )
    elif disqualified_by_critical_skills:
        missing_critical_text = ', '.join(missing_critical_skills) or "one or more critical skills"
        opening = (
            f"This candidate has been automatically disqualified for the {job_title} role "
            f"because they are missing critical skill(s): {missing_critical_text}. "
            f"The overall fit score remains {fit_score:.0f}% to show the weighted evidence separately."
        )
    elif tier == "strong":
        opening = (
            f"This candidate is a strong match for the {job_title} role, "
            f"achieving an overall fit score of {fit_score:.0f}% - well above the qualification threshold."
        )
    elif tier == "moderate":
        opening = (
            f"This candidate shows a moderate fit for the {job_title} role "
            f"with an overall fit score of {fit_score:.0f}%, placing them in the 'For Review' category."
        )
    else:
        opening = (
            f"This candidate does not meet the minimum qualifications for the {job_title} role, "
            f"scoring {fit_score:.0f}% overall - below the required threshold."
        )

    # --- Skills section ---
    if total_required == 0:
        skills_text = "No specific required skills were defined for this role, so skill matching was not evaluated."
    elif matched_count == total_required:
        skills_text = (
            f"Notably, the candidate demonstrates all {total_required} required skill(s) "
            f"({', '.join(matched_skills)}), achieving a perfect skills match score."
        )
    elif matched_count == 0:
        skills_text = (
            f"The resume did not surface any of the {total_required} required skill(s) "
            f"({', '.join(missing_skills)}). This is the primary driver of the low overall score."
        )
    else:
        matched_str = ', '.join(matched_skills[:5]) + (f" and {matched_count - 5} more" if matched_count > 5 else "")
        missing_str = ', '.join(missing_skills[:5]) + (f" and {missing_count - 5} more" if missing_count > 5 else "")
        preferred_note = ""
        if matched_preferred:
            preferred_note = f" The candidate also matched {len(matched_preferred)} preferred skill(s) ({', '.join(matched_preferred[:3])}{'...' if len(matched_preferred) > 3 else ''}), earning a +{preferred_bonus:.0f}pt bonus."
        skills_text = (
            f"Out of {total_required} required skill(s), {matched_count} matched: {matched_str}. "
            f"The {missing_count} missing skill(s) - {missing_str} - account for the skills gap and "
            f"result in a required skills score of {skill_score:.0f}%.{preferred_note}"
        )

    if matched_critical_skills or missing_critical_skills:
        critical_parts = []
        if matched_critical_skills:
            critical_parts.append(f"matched critical skill(s): {', '.join(matched_critical_skills)}")
        if missing_critical_skills:
            critical_parts.append(f"missing critical skill(s): {', '.join(missing_critical_skills)}")
        skills_text = f"{skills_text} Critical-skill check: {'; '.join(critical_parts)}."

    # --- Experience section ---
    if experience_req and experience_req > 0:
        if total_exp_years >= experience_req:
            exp_text = (
                f"Their work history indicates approximately {total_exp_years:.1f} year(s) of relevant experience, "
                f"meeting or exceeding the {experience_req}-year requirement (experience score: {exp_score:.0f}%)."
            )
        elif total_exp_years > 0:
            exp_text = (
                f"Their work history reflects around {total_exp_years:.1f} year(s) of experience, "
                f"which falls short of the {experience_req}-year requirement "
                f"(experience score: {exp_score:.0f}%)."
            )
        else:
            exp_text = (
                f"No verifiable work experience timeline was detected in the resume. "
                f"The role requires {experience_req} year(s) of experience, resulting in an experience score of 0%."
            )
    else:
        if total_exp_years > 0:
            exp_text = (
                f"No minimum experience was specified for this role; the candidate's resume indicates "
                f"approximately {total_exp_years:.1f} year(s) of relevant work history."
            )
        else:
            exp_text = (
                "No minimum experience was specified for this role, and no explicit experience timeline "
                "was detected in the resume."
            )

    # --- Education section ---
    cand_edu_rank = 0
    if extracted_edu:
        cand_edu_rank = max(get_degree_rank(e['degree']) for e in extracted_edu)
    cand_edu_label = _degree_label(cand_edu_rank)
    job_edu_rank = get_degree_rank(education_req)
    job_edu_label = _degree_label(job_edu_rank) if job_edu_rank > 0 else None

    if not job_edu_label:
        edu_text = (
            f"No specific education requirement was set for this role. "
            f"The candidate's highest detected credential is a {cand_edu_label}."
        )
    elif cand_edu_rank >= job_edu_rank:
        edu_text = (
            f"The candidate's detected education level ({cand_edu_label}) meets or exceeds "
            f"the required {job_edu_label} for this role (education score: {edu_score:.0f}%)."
        )
    else:
        edu_text = (
            f"The candidate's detected education ({cand_edu_label}) does not meet "
            f"the required {job_edu_label} for this role (education score: {edu_score:.0f}%)."
        )

    # --- Closing recommendation ---
    if disqualification_reason or disqualified_by_critical_skills:
        closing = (
            "Recommendation: Do not advance unless the rule-triggering gap is resolved or verified differently "
            "during human review."
        )
    elif tier == "strong":
        closing = (
            "Recommendation: Advance to the next stage. This candidate presents a compelling profile "
            "and is well-suited for further evaluation."
        )
    elif tier == "moderate":
        closing = (
            "Recommendation: Human review advised. The candidate shows promise but has gaps that should "
            "be explored further during an interview or assessment."
        )
    else:
        closing = (
            "Recommendation: Not advised for advancement at this stage. The candidate's profile "
            "does not meet the minimum criteria required for this position."
        )

    return " ".join([opening, skills_text, exp_text, edu_text, closing])


def generate_decision_explanation(job_title, recommendation_label, fit_score,
                                  skill_score, exp_score, edu_score,
                                  text_similarity_score, matched_skills,
                                  missing_skills, matched_critical_skills,
                                  missing_critical_skills, matched_preferred,
                                  preferred_bonus, total_exp_years,
                                  experience_req, education_req):
    """
    Builds a concise explanation of the final screening decision.
    It keeps the key evidence, risks, scoring weights, and reviewer action visible.
    """
    matched_skills = matched_skills or []
    missing_skills = missing_skills or []
    matched_critical_skills = matched_critical_skills or []
    missing_critical_skills = missing_critical_skills or []
    matched_preferred = matched_preferred or []

    total_required = len(matched_skills) + len(missing_skills)
    matched_count = len(matched_skills)
    required_ratio = (matched_count / total_required) if total_required else 1.0
    seed = "|".join([
        job_title,
        recommendation_label,
        f"{fit_score:.2f}",
        ",".join(matched_skills),
        ",".join(missing_skills),
        f"{total_exp_years:.1f}",
    ])

    if recommendation_label == "Qualified":
        verdict_text = _pick_phrase([
            f"Decision: Qualified for {job_title}.",
            f"Decision: This applicant is qualified for {job_title}.",
            f"Decision: The resume reached Qualified status for {job_title}.",
        ], seed)
        verdict_text += f" Fit score: {fit_score:.0f}%. No hard disqualification rule was triggered."
    elif recommendation_label == "For Review":
        verdict_text = _pick_phrase([
            f"Decision: For Review for {job_title}.",
            f"Decision: This applicant needs manual review for {job_title}.",
            f"Decision: The resume is not a clear pass or fail for {job_title}.",
        ], seed)
        verdict_text += f" Fit score: {fit_score:.0f}%. Some useful evidence was found, but important gaps still need checking."
    else:
        verdict_text = _pick_phrase([
            f"Decision: Not Qualified for {job_title}.",
            f"Decision: This applicant is not qualified for {job_title}.",
            f"Decision: The resume did not meet the minimum screening standard for {job_title}.",
        ], seed)
        verdict_text += f" Fit score: {fit_score:.0f}%. One or more screening rules found a gap that is too large to ignore."

    strengths = []
    if matched_skills:
        strengths.append(f"required skill(s): {', '.join(matched_skills[:5])}")
    if matched_critical_skills:
        strengths.append(f"critical skill(s): {', '.join(matched_critical_skills[:5])}")
    if matched_preferred:
        strengths.append(f"preferred skill(s): {', '.join(matched_preferred[:5])}")
    if total_exp_years > 0:
        if experience_req and experience_req > 0:
            if total_exp_years >= experience_req:
                strengths.append(
                    f"work experience advantage: about {total_exp_years:.1f} year(s), meeting the {experience_req}-year requirement"
                )
            else:
                strengths.append(
                    f"work experience advantage: about {total_exp_years:.1f} year(s), but below the {experience_req}-year requirement"
                )
        else:
            strengths.append(f"work experience advantage: about {total_exp_years:.1f} year(s)")
    elif not experience_req or experience_req <= 0:
        strengths.append("no minimum experience requirement was set")
    if edu_score >= 100.0:
        strengths.append("education requirement appears satisfied")

    if strengths:
        strengths_text = _pick_phrase([
            f"Strengths: the applicant's main advantage is {strengths[0]}.",
            f"Strengths: the strongest evidence is {strengths[0]}.",
            f"Strengths: what works in the applicant's favor is {strengths[0]}.",
        ], seed, 1)
        if len(strengths) > 1:
            strengths_text += f" Also found: {'; '.join(strengths[1:])}."
    else:
        strengths_text = "Strengths: no clear advantage was detected from the configured requirements."

    weaknesses = []
    if missing_critical_skills:
        weaknesses.append(f"missing critical skill(s): {', '.join(missing_critical_skills)}")
    if missing_skills:
        weaknesses.append(f"missing required skill(s): {', '.join(missing_skills)}")
    if experience_req and experience_req > 0 and exp_score < 100.0:
        weaknesses.append(f"experience score did not fully clear the {experience_req}-year requirement")
    if edu_score < 100.0:
        weaknesses.append(f"education did not fully match the configured requirement ({education_req or 'not specified'})")

    if weaknesses:
        weaknesses_text = _pick_phrase([
            f"Concern: {weaknesses[0]}.",
            f"Gap: {weaknesses[0]}.",
            f"Weakness: {weaknesses[0]}.",
        ], seed, 2)
        if len(weaknesses) > 1:
            weaknesses_text += f" Also check: {'; '.join(weaknesses[1:])}."
    else:
        weaknesses_text = "Concern: no major gap was detected, but the extracted resume details should still be verified."

    risk_points = []

    if missing_critical_skills:
        risk_points.append(f"missing critical skill(s): {', '.join(missing_critical_skills)}")
    if total_required and len(matched_skills) == 0:
        risk_points.append("zero required skills were matched")
    elif total_required and required_ratio < 0.5:
        risk_points.append("less than half of the required skills were matched")
    if experience_req and experience_req > 0 and exp_score < 50.0:
        risk_points.append("experience is far below the minimum requirement")
    elif experience_req and experience_req > 0 and exp_score < 100.0:
        risk_points.append("experience is below the stated minimum requirement")
    if missing_skills and recommendation_label != "Not Qualified":
        risk_points.append("some required skills are still missing")

    risk_text = (
        _pick_phrase([
            f"Risk to review: {', '.join(risk_points)}.",
            f"Manual review should focus on: {', '.join(risk_points)}.",
            f"Most important risk: {', '.join(risk_points)}.",
        ], seed, 3)
        if risk_points else
        _pick_phrase([
            "Risk to review: no major rule-based risk was detected.",
            "Manual review focus: verify that the extracted resume details are accurate.",
            "Most important risk: no high-risk issue was detected by the configured rules.",
        ], seed, 4)
    )

    if total_required:
        requirements_text = (
            f"Matched {matched_count} out of {total_required} required skill(s). "
            f"Scores: skills {skill_score:.0f}%, experience {exp_score:.0f}%, education {edu_score:.0f}%."
        )
    else:
        requirements_text = (
            f"No required-skill list was provided. "
            f"Scores: experience {exp_score:.0f}%, education {edu_score:.0f}%."
        )

    preferred_text = (
        f"Preferred skill bonus: +{preferred_bonus:.2f}; it does not replace required skills."
        if matched_preferred else
        "No preferred-skill bonus was applied."
    )
    similarity_text = (
        f"Text similarity: {text_similarity_score:.0f}% reference metric only."
        if text_similarity_score is not None else
        "Text similarity was not available and did not affect the decision."
    )
    scoring_text = (
        _pick_phrase([
            "Scoring: skills 50%, experience 30%, education 20%.",
            "Score basis: skills 50%, experience 30%, education 20%.",
            "Fit score weights: skills 50%, experience 30%, education 20%.",
        ], seed, 5)
        + f" {requirements_text} {preferred_text} {similarity_text}"
    )

    if recommendation_label == "Qualified":
        reviewer_text = _pick_phrase([
            "Reviewer recommendation: advance to the next screening stage after verifying the extracted evidence.",
            "Reviewer recommendation: consider moving forward, but confirm the resume details manually.",
            "Reviewer recommendation: suitable for next-stage screening, pending human verification.",
        ], seed, 6)
    elif recommendation_label == "For Review":
        reviewer_text = _pick_phrase([
            "Reviewer recommendation: review the gaps before deciding whether to move forward.",
            "Reviewer recommendation: check the weaker areas manually before advancing.",
            "Reviewer recommendation: use this as a review queue item, not a final decision.",
        ], seed, 7)
    else:
        reviewer_text = _pick_phrase([
            "Reviewer recommendation: do not advance unless clearer evidence addresses the missing requirements.",
            "Reviewer recommendation: hold from advancement unless additional information resolves the gaps.",
            "Reviewer recommendation: current resume evidence is not strong enough for next-stage screening.",
        ], seed, 8)

    return "\n\n".join([
        verdict_text,
        strengths_text,
        weaknesses_text,
        scoring_text,
        f"{risk_text} {reviewer_text}",
    ])


def estimate_decision_confidence(recommendation_label, resume_text, contact_info,
                                 extracted_edu, extracted_exp, total_exp_years,
                                 required_skills, matched_skills, missing_skills,
                                 matched_critical_skills, missing_critical_skills,
                                 skill_score, exp_score, edu_score, fit_score,
                                 experience_req=0, education_req=None,
                                 min_fit_score=50.0):
    """
    Estimates how much trust the system should place in its own screening result.
    This is not a second decision; it tells the reviewer whether the extracted
    evidence is strong enough to trust or should be checked manually.
    """
    resume_text = resume_text or ""
    contact_info = contact_info or {}
    extracted_edu = extracted_edu or []
    extracted_exp = extracted_exp or []
    required_skills = [skill for skill in (required_skills or []) if str(skill).strip()]
    matched_skills = matched_skills or []
    missing_skills = missing_skills or []
    matched_critical_skills = matched_critical_skills or []
    missing_critical_skills = missing_critical_skills or []

    # Confidence is built from independently observable evidence.  Starting at
    # zero prevents a sparse or failed extraction from looking trustworthy just
    # because the resulting fit score is far from a decision threshold.
    score = 0
    strengths = []
    concerns = []
    text_length = len(resume_text.strip())
    words = _re.findall(r"[A-Za-z][A-Za-z'-]+", resume_text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0.0
    nonempty_lines = sum(bool(line.strip()) for line in resume_text.splitlines())

    # Length alone is easy to inflate (for example, repeated boilerplate), so
    # lexical variety and line structure also contribute to extraction quality.
    if text_length >= 700 and len(words) >= 100 and unique_ratio >= 0.18 and nonempty_lines >= 8:
        score += 25
        strengths.append("resume text is substantial and structurally readable")
    elif text_length >= 300 and len(words) >= 45 and unique_ratio >= 0.12:
        score += 16
        strengths.append("resume text is readable but has limited extraction coverage")
    elif text_length > 0:
        score += 4
        concerns.append("resume text is sparse or repetitive, so extraction may be incomplete")
    else:
        concerns.append("no readable resume text was available")

    candidate_name = (contact_info.get("name") or "").strip().lower()
    if candidate_name and candidate_name not in {"unknown candidate", "unknown", "n/a"}:
        score += 7
        strengths.append("candidate identity was detected")
    else:
        concerns.append("candidate name was not confidently detected")

    if contact_info.get("email") or contact_info.get("phone"):
        score += 3
        strengths.append("contact evidence was detected")

    total_required = len(required_skills)
    matched_required = len(matched_skills)
    required_ratio = matched_required / total_required if total_required else 1.0
    if total_required:
        score += 15
        if matched_required + len(missing_skills) == total_required:
            score += 5
            strengths.append("every configured required skill was evaluated")
        else:
            concerns.append("required-skill results do not reconcile with the configured list")

        # A missing-skill conclusion is only strong when the source extraction
        # itself is healthy; absence in sparse text is not reliable evidence.
        if required_ratio == 1.0:
            score += 5
            strengths.append("all required skills have direct text evidence")
        elif required_ratio == 0.0 and score < 40:
            concerns.append("missing skills may reflect incomplete text extraction")
    else:
        concerns.append("no required-skill list was configured")

    if missing_critical_skills:
        if text_length >= 300 and len(words) >= 45:
            score += 5
            strengths.append("critical-skill rule was evaluated against readable text")
        else:
            concerns.append("critical skills appear missing from incomplete resume text")
    elif matched_critical_skills:
        score += 5
        strengths.append("critical-skill evidence was found")

    if experience_req and experience_req > 0:
        if extracted_exp or total_exp_years > 0:
            score += 10
            strengths.append("work experience evidence was detected")
        else:
            concerns.append("work experience requirement exists but no work history was extracted")

        complete_exp = sum(
            bool(record.get("job_title")) and bool(record.get("company"))
            for record in extracted_exp if isinstance(record, dict)
        )
        if complete_exp:
            score += min(8, complete_exp * 3)
        elif extracted_exp:
            concerns.append("work-history records are missing a title or employer")

        record_years = sum(max(0, float(record.get("years") or 0)) for record in extracted_exp if isinstance(record, dict))
        if record_years and total_exp_years and abs(record_years - total_exp_years) > max(2.0, total_exp_years * 0.5):
            score -= 8
            concerns.append("total experience conflicts with extracted work-history durations")
        elif record_years and total_exp_years:
            score += 4
            strengths.append("experience totals agree with extracted work history")

        if 45.0 <= exp_score <= 55.0 or 90.0 <= exp_score < 100.0:
            score -= 5
            concerns.append("experience score is near a review boundary")
    elif extracted_exp or total_exp_years > 0:
        strengths.append("work history was detected")

    if education_req and get_degree_rank(education_req) > 0:
        if extracted_edu or edu_score >= 100.0:
            score += 8
            strengths.append("education evidence was detected")
        else:
            concerns.append("education requirement exists but no education history was extracted")

        complete_edu = any(
            isinstance(record, dict) and record.get("degree") and record.get("institution")
            for record in extracted_edu
        )
        if complete_edu:
            score += 4
        elif extracted_edu:
            concerns.append("education record is missing a degree or institution")

        if 80.0 <= edu_score < 100.0:
            score -= 4
            concerns.append("education score is close but not a full match")
    elif extracted_edu:
        strengths.append("education history was detected")

    threshold_distance = min(abs(fit_score - 75.0), abs(fit_score - min_fit_score))
    if threshold_distance <= 3.0:
        concerns.append("fit score is close to a decision threshold")
    elif threshold_distance >= 12.0:
        score += 10
        strengths.append("fit score is well separated from decision thresholds")
    else:
        score += 5

    if recommendation_label == "For Review":
        concerns.append("decision is intentionally routed for human review")

    score = max(0, min(100, round(score)))

    if score >= 75:
        level = "High"
    elif score >= 45:
        level = "Medium"
    else:
        level = "Low"

    if recommendation_label == "For Review" and level == "High":
        level = "Medium"

    reason_parts = [f"Evidence confidence score: {score}/100."]
    if strengths:
        reason_parts.append("Confidence support: " + "; ".join(strengths[:3]) + ".")
    if concerns:
        reason_parts.append("Reviewer should verify: " + "; ".join(concerns[:3]) + ".")
    if not reason_parts:
        reason_parts.append("Confidence is based on the available screening signals.")

    return level, " ".join(reason_parts)


def evaluate_candidate(resume_text, job_desc_text, required_skills, min_fit_score=50.0,
                       experience_req=0, education_req=None, requires_all_critical=False,
                       job_title="the target role", preferred_skills=None,
                       critical_skills=None):
    """
    Orchestrates the evaluation of a candidate's resume against a job description.
    Returns a dictionary with scores and recommendation.
    """
    from .nlp_pipeline import (
        clean_text, extract_contact_info, extract_education, 
        extract_years_of_experience, extract_experience_records, extract_certifications
    )
    from .matching_engine import calculate_fit_score, calculate_text_similarity
    
    # 1. Clean Texts
    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_desc_text)

    # 2. Required skill matching — run against original resume text (not cleaned) so that
    #    versioned skills like 'CSS3', 'ES6', 'HTML5', 'Python 3' are not stripped of digits.
    matched_skills, missing_skills = analyze_skills(resume_text, required_skills)
    matched_critical_skills, missing_critical_skills = analyze_skills(
        resume_text, critical_skills or []
    )

    # 2b. Preferred skill bonus — also uses original text for the same reason
    matched_preferred, total_preferred, preferred_bonus = analyze_preferred_skills(
        resume_text, preferred_skills or []
    )

    # 3. Extract contact info, education, experience
    contact_info = extract_contact_info(resume_text)
    extracted_edu = extract_education(resume_text)
    extracted_exp = extract_experience_records(resume_text)
    extracted_certifications = extract_certifications(resume_text)
    total_exp_years = extract_years_of_experience(resume_text)

    # 4. Required Skills Match Score (0–100), then apply preferred bonus (capped at 100)
    if required_skills:
        required_skill_score = (len(matched_skills) / len(required_skills)) * 100.0
    else:
        required_skill_score = 100.0
    skill_score = min(required_skill_score + preferred_bonus, 100.0)

    # 5. Experience Score with soft ceiling — extra years beyond 2× the requirement
    #    give diminishing returns rather than capping hard at 100%
    if not experience_req or experience_req <= 0:
        exp_score = 100.0
    else:
        raw_ratio = total_exp_years / experience_req
        if raw_ratio >= 1.0:
            # Met requirement — scale from 100% up, but cap at 100
            exp_score = 100.0
        else:
            # Partial credit; scale linearly up to 100%
            exp_score = round(raw_ratio * 100.0, 2)

    # 6. Education Score
    cand_rank = 0
    if extracted_edu:
        cand_rank = max(get_degree_rank(e['degree']) for e in extracted_edu)
    else:
        # Fallback: scan raw text directly
        cand_rank = get_degree_rank(resume_text)

    job_rank = get_degree_rank(education_req)
    if job_rank <= 0:
        edu_score = 100.0
    else:
        edu_score = min((cand_rank / job_rank) * 100.0, 100.0)

    # 7. Final Weighted Fit Score (3-component model: Skills 50%, Experience 30%, Education 20%)
    fit_score = calculate_fit_score(skill_score, exp_score, edu_score)
    text_similarity_score = calculate_text_similarity(cleaned_resume, cleaned_job)
    
    # 9. Recommendation & Critical Skill Enforcement
    label = generate_recommendation(fit_score, min_fit_score)
    disqualified_by_critical_skills = False
    disqualification_reason = None

    if requires_all_critical and missing_critical_skills:
        label = "Not Qualified"
        disqualified_by_critical_skills = True
        missing_critical_text = ', '.join(missing_critical_skills)
        disqualification_reason = f"the resume is missing required critical skill(s): {missing_critical_text}"
    elif required_skills and len(matched_skills) == 0:
        label = "Not Qualified"
        disqualification_reason = "none of the configured required skills were found in the resume"
    elif experience_req and experience_req > 0 and exp_score < 50.0:
        label = "Not Qualified"
        disqualification_reason = "the detected work experience is far below the configured minimum requirement"
    elif experience_req and experience_req > 0 and exp_score < 100.0 and label == "Qualified":
        label = "For Review"

    # 9. Generate Unique Candidate-Specific Narrative
    summary = generate_analysis_narrative(
        job_title=job_title,
        fit_score=fit_score,
        skill_score=skill_score,
        exp_score=exp_score,
        edu_score=edu_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        total_exp_years=total_exp_years,
        extracted_edu=extracted_edu,
        experience_req=experience_req,
        education_req=education_req,
        disqualified_by_critical_skills=disqualified_by_critical_skills,
        matched_preferred=matched_preferred,
        preferred_bonus=preferred_bonus,
        matched_critical_skills=matched_critical_skills,
        missing_critical_skills=missing_critical_skills,
        disqualification_reason=disqualification_reason
    )
    decision_explanation = generate_decision_explanation(
        job_title=job_title,
        recommendation_label=label,
        fit_score=fit_score,
        skill_score=skill_score,
        exp_score=exp_score,
        edu_score=edu_score,
        text_similarity_score=text_similarity_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        matched_critical_skills=matched_critical_skills,
        missing_critical_skills=missing_critical_skills,
        matched_preferred=matched_preferred,
        preferred_bonus=preferred_bonus,
        total_exp_years=total_exp_years,
        experience_req=experience_req,
        education_req=education_req
    )
    confidence_level, confidence_reason = estimate_decision_confidence(
        recommendation_label=label,
        resume_text=resume_text,
        contact_info=contact_info,
        extracted_edu=extracted_edu,
        extracted_exp=extracted_exp,
        total_exp_years=total_exp_years,
        required_skills=required_skills,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        matched_critical_skills=matched_critical_skills,
        missing_critical_skills=missing_critical_skills,
        skill_score=skill_score,
        exp_score=exp_score,
        edu_score=edu_score,
        fit_score=fit_score,
        experience_req=experience_req,
        education_req=education_req,
        min_fit_score=min_fit_score,
    )
    return {
        "skill_score": round(skill_score, 2),
        "experience_score": round(exp_score, 2),
        "education_score": round(edu_score, 2),
        "text_similarity_score": text_similarity_score,
        "fit_score": round(fit_score, 2),
        "recommendation_label": label,
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_critical_skills": matched_critical_skills,
        "missing_critical_skills": missing_critical_skills,
        "matched_preferred": matched_preferred,
        "summary": summary,
        "decision_explanation": decision_explanation,
        "contact_info": contact_info,
        "extracted_edu": extracted_edu,
        "extracted_exp": extracted_exp,
        "extracted_certifications": extracted_certifications,
        "total_exp_years": total_exp_years
    }