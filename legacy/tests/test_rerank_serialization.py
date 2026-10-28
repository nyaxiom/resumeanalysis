from resumeanalysis_legacy.rerank.evidence import cross_encoder_texts

from resumeanalysis.schemas import Profile, Vacancy


def test_serializer_preserves_supplied_text_and_structured_metadata_excludes_title():
    profile = Profile(resume_text="Python / разработка", experience_years=3, seniority="middle")
    job = Vacancy(
        title="SecretTitleSkill",
        description="Python needed",
        min_experience_years=2,
        seniority="junior",
    )
    resume, vacancy = cross_encoder_texts(profile, job)
    assert "Python / разработка" in resume
    assert "Stated experience: 3 years" in resume
    assert "Required seniority: junior" in vacancy
    assert "SecretTitleSkill" not in vacancy
