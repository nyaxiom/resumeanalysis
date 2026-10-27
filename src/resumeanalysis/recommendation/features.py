"""Compact, versioned features reusing frozen E5 and existing skill evidence."""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.random_projection import GaussianRandomProjection

from resumeanalysis.extraction.registry import create_extractor, extractor_identity
from resumeanalysis.features.feature_builder import profile_text, vacancy_text
from resumeanalysis.recommendation.adapters import matching_facts
from resumeanalysis.recommendation.constraints import assess_constraints, salary_compatibility
from resumeanalysis.recommendation.schema import (
    ConstraintStatus,
    FeatureSnapshot,
    QualificationEvidence,
    fingerprint,
)
from resumeanalysis.rerank.evidence import build_evidence


@dataclass
class BuiltFeatures:
    snapshot: FeatureSnapshot
    qualification: QualificationEvidence
    constraints: ConstraintStatus


class PreferenceFeatureBuilder:
    def __init__(self, encoder, *, embedding_components=0, extractor=None):
        if embedding_components not in {0, 16}:
            raise ValueError("Choose compact features or the evaluated 16-component alternative")
        self.encoder = encoder
        self.extractor = extractor if extractor is not None else create_extractor("v2")
        self.schema_version = f"preference-features-v1-rp{embedding_components}"
        self.identity = fingerprint(
            {
                "schema": self.schema_version,
                "encoder": encoder.identity,
                "dimension": encoder.dimension,
                "extractor": extractor_identity(self.extractor),
                "taxonomy": self.extractor.normalizer.taxonomy.identity,
                "projection_seed": 42,
                "category_hash": "sklearn-murmur32-signed-8-per-field-v1",
            }
        )
        self.projector = None
        self.category_hasher = FeatureHasher(n_features=8, input_type="string", alternate_sign=True)
        if embedding_components:
            # GaussianRandomProjection draws a fixed matrix; this fits no training examples.
            self.projector = GaussianRandomProjection(
                n_components=embedding_components, random_state=42
            ).fit(np.zeros((1, encoder.dimension)))

    def _encode(self, text, role):
        vector = np.asarray(self.encoder.encode(text, role), dtype=np.float64)
        if vector.shape != (self.encoder.dimension,) or not np.isfinite(vector).all():
            raise ValueError("Encoder returned an invalid recommendation vector")
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def build(self, profile, vacancy) -> BuiltFeatures:
        facts = matching_facts(profile)
        evidence = build_evidence(facts, vacancy.facts, self.extractor)
        detail, metadata = evidence.detail, evidence.metadata
        confirmed, denied = set(metadata["confirmed_skills"]), set(metadata["denied_skills"])
        features = {}

        def known(name, value):
            features[name] = float(value) if value is not None else 0.0
            features[name + "_known"] = float(value is not None)

        for category, skills in (("required", detail.required), ("optional", detail.optional)):
            units = [{s} for s in sorted(skills)] + [
                set(group["skills"])
                for group in detail.alternative_groups
                if group["requirement"] == category
            ]
            known(
                category + "_coverage",
                sum(bool(unit & confirmed) for unit in units) / len(units)
                if units and (confirmed or denied)
                else None,
            )
            features[category + "_requirements_known"] = float(bool(units))
            features[category + "_unknown"] = (
                sum(not (u & confirmed) and not (u <= denied) for u in units) / len(units)
                if units
                else 0.0
            )
            features[category + "_denied"] = (
                sum(u <= denied for u in units) / len(units) if units else 0.0
            )
        features["candidate_skills_known"] = float(bool(confirmed or denied))
        known(
            "related_support",
            sum(
                self.extractor.normalizer.taxonomy.related_support(confirmed, s)
                for s in detail.required - confirmed
            )
            / len(detail.required)
            if detail.required
            else None,
        )
        ptext, jtext = profile_text(facts), vacancy_text(vacancy.facts)
        candidate_present = bool(
            facts.resume_text.strip() or facts.skills or facts.projects or facts.education
        )
        job_present = bool(
            vacancy.facts.description.strip()
            or vacancy.facts.required_skills
            or vacancy.facts.optional_skills
        )
        job_vector = self._encode(jtext, "passage") if job_present else None
        similarity = None
        if candidate_present and job_vector is not None:
            similarity = float(np.clip(np.dot(self._encode(ptext, "query"), job_vector), -1, 1))
        known("semantic_similarity", similarity)
        years, minimum = profile.facts.experience_years, vacancy.facts.min_experience_years
        gap = max(0.0, minimum - years) if years is not None and minimum is not None else None
        known("experience_match", float(gap == 0) if gap is not None else None)
        known("experience_gap", min(gap, 20) / 20 if gap is not None else None)
        pref = profile.preferences
        known("salary_match", salary_compatibility(pref.salary, vacancy.salary))
        known(
            "remote_match",
            float(pref.remote == vacancy.facts.remote)
            if pref.remote is not None and vacancy.facts.remote is not None
            else None,
        )
        known(
            "location_match",
            float(
                vacancy.facts.location.casefold() in {value.casefold() for value in pref.locations}
            )
            if pref.locations and vacancy.facts.location
            else None,
        )
        roles = pref.preferred_roles
        role_match = None
        if roles and vacancy.facts.title.strip():
            title = self._encode(vacancy.facts.title, "passage")
            role_match = max(
                float(np.clip(np.dot(self._encode(role, "query"), title), -1, 1)) for role in roles
            )
        known("role_similarity", role_match)
        # Absolute vacancy traits let feedback discover preferences that the user
        # has never stated. Compatibility-only features cannot learn this signal.
        known(
            "vacancy_remote",
            float(vacancy.facts.remote) if vacancy.facts.remote is not None else None,
        )
        for name, value in (("role", vacancy.facts.title), ("location", vacancy.facts.location)):
            tokens = [" ".join(value.casefold().split())] if value and value.strip() else []
            hashed = self.category_hasher.transform([tokens]).toarray()[0]
            features.update({f"{name}_hash_{i}": float(v) for i, v in enumerate(hashed)})
            features[f"vacancy_{name}_known"] = float(bool(tokens))
        if self.projector is not None:
            projected = (
                self.projector.transform(job_vector.reshape(1, -1))[0]
                if job_vector is not None
                else np.zeros(16)
            )
            features.update({f"vacancy_rp_{i:02d}": float(v) for i, v in enumerate(projected)})
            features["vacancy_embedding_known"] = float(job_vector is not None)
        if not all(np.isfinite(value) for value in features.values()):
            raise ValueError("Nonfinite preference features")

        # These are transparent cold-start ranking heuristics, not trained/calibrated probabilities.
        qualification_terms = []
        for name, weight in (
            ("required_coverage", 0.55),
            ("optional_coverage", 0.15),
            ("semantic_similarity", 0.2),
            ("experience_match", 0.1),
        ):
            if features[name + "_known"]:
                value = features[name]
                qualification_terms.append((max(0.0, value), weight))
        qualification = (
            (sum(v * w for v, w in qualification_terms) / sum(w for _, w in qualification_terms))
            if qualification_terms
            else None
        )
        content_terms = [(qualification, 0.6)] if qualification is not None else []
        for name, weight in (
            ("role_similarity", 0.2),
            ("salary_match", 0.1),
            ("remote_match", 0.05),
            ("location_match", 0.05),
        ):
            if features[name + "_known"]:
                content_terms.append((max(0.0, features[name]), weight))
        content = (
            sum(v * w for v, w in content_terms) / sum(w for _, w in content_terms)
            if content_terms
            else 0.5
        )
        return BuiltFeatures(
            FeatureSnapshot(
                schema_version=self.schema_version,
                identity=self.identity,
                values=dict(sorted(features.items())),
                profile_id=profile.profile_id,
                profile_version=profile.version,
                profile_sha256=fingerprint(profile),
                vacancy_sha256=fingerprint(vacancy),
                content_score=float(content),
            ),
            QualificationEvidence(
                score=qualification,
                matched_skills=sorted(detail.matched),
                missing_skills=sorted(detail.unknown | detail.contradicted),
                unknown_skills=sorted(detail.unknown),
                contradicted_skills=sorted(detail.contradicted),
                experience_gap_years=gap,
                alternative_requirements=detail.alternative_groups,
                explanation="Qualification evidence is independent of likes/dislikes. Missing or unknown skills are unconfirmed, not proof of incapacity. "
                + " ".join(evidence.blocked_reasons + evidence.review_reasons),
            ),
            assess_constraints(profile, vacancy, detail),
        )
