"""Canonical feature definitions for the resume-screening proxy-bias experiment.

Single source of truth for the model/similarity feature set, imported by
notebooks 02-06 so the list is defined exactly once. Replaces the three
verbatim copies (02, 05, 06) and the two exclude-rule derivations (03, 04)
that previously drifted independently.

The label internals (``score_*``), the protected attributes (``true_age``,
``age_group``), identifiers, and the downstream-outcome columns are NOT
features and must never enter a model input or similarity space.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Display/aggregation order for the five age bands. Shared by every notebook
# (02-08); previously hand-typed per notebook, where one typo would silently
# reindex a table to NaN rows.
# ---------------------------------------------------------------------------
AGE_ORDER: list[str] = ["<30", "30-39", "40-49", "50-59", "60+"]

# ---------------------------------------------------------------------------
# The 51 screening features (order is stable and matches notebooks 02/05/06).
# ---------------------------------------------------------------------------
CANONICAL_FEATURES: list[str] = [
    "application_year", "target_role_family", "target_role_level", "region",
    "highest_degree", "graduation_year", "school_tier", "gpa_bucket",
    "years_experience_total", "years_experience_relevant", "num_employers",
    "avg_tenure_years", "months_since_last_role", "num_gaps_over_6mo",
    "most_recent_title", "most_recent_company_size", "management_years", "reports_max",
    "num_skills_listed", "num_programming_languages", "num_cloud_platforms", "num_databases",
    "skill_python", "skill_java", "skill_javascript", "skill_go", "skill_kubernetes",
    "skill_aws", "skill_gcp", "skill_azure", "skill_sql", "skill_spark",
    "skill_terraform", "skill_linux", "skill_ml", "legacy_tech_count", "modern_tech_count",
    "cert_count", "has_top_cloud_cert", "github_url_present", "portfolio_url_present",
    "open_source_mentions", "patent_count", "resume_word_count", "bullet_count",
    "quantified_impact_count", "keyword_match_score", "format_clean_score",
    "salary_expectation_usd", "willing_to_relocate", "remote_only",
]

# Convenience: features minus the most direct date proxy (notebooks 04/05/06).
FEATURES_WITHOUT_GRAD_YEAR: list[str] = [c for c in CANONICAL_FEATURES if c != "graduation_year"]

# ---------------------------------------------------------------------------
# The age-proxy cluster (notebook 04): features that are near-deterministic
# functions of age in the generator. Removing the whole cluster collapses the
# disparity; removing any single member does not (mutual substitutability).
# ---------------------------------------------------------------------------
PROXY_CLUSTER: list[str] = [
    "graduation_year", "years_experience_total", "years_experience_relevant",
    "num_employers", "avg_tenure_years", "management_years", "reports_max",
    "most_recent_title", "legacy_tech_count", "modern_tech_count", "salary_expectation_usd",
]

# Experience columns, for the notebook-05 drop-experience arm (punch-list #4).
EXPERIENCE_FEATURES: list[str] = [
    "years_experience_total", "years_experience_relevant",
]

# ---------------------------------------------------------------------------
# Non-feature columns. Anything here (plus any ``score_*``) is excluded from
# every feature set. Used by the validator to catch schema drift.
# ---------------------------------------------------------------------------
PROTECTED_COLUMNS: frozenset[str] = frozenset({"true_age", "age_group"})
IDENTIFIER_COLUMNS: frozenset[str] = frozenset({"candidate_id"})
OUTCOME_COLUMNS: frozenset[str] = frozenset({"callback", "interview", "offer"})
# Latent generator internals that are not part of the rendered resume features.
LATENT_COLUMNS: frozenset[str] = frozenset({
    "estimated_start_year", "tech_recency_score", "leadership_signal_score", "stability_score",
})

NON_FEATURE_COLUMNS: frozenset[str] = (
    PROTECTED_COLUMNS | IDENTIFIER_COLUMNS | OUTCOME_COLUMNS | LATENT_COLUMNS
)


def is_score_column(name: str) -> bool:
    """Label internals are stored as ``score_*`` and are never features."""
    return name.startswith("score_")


def feature_columns(df) -> list[str]:
    """Return the canonical feature columns present in ``df`` (order preserved).

    This is the drop-in replacement for the per-notebook exclude-rule:
    everything that is not a score column, protected attribute, identifier,
    outcome, or latent internal.
    """
    return [c for c in CANONICAL_FEATURES if c in df.columns]


def validate_features(df, *, strict: bool = True) -> list[str]:
    """Assert the dataframe's feature columns are exactly ``CANONICAL_FEATURES``.

    Recomputes the feature set from ``df`` by exclusion and checks it matches
    the canonical list. This is the single-source guard: if the generator's
    schema ever changes, this trips loudly instead of letting a stale literal
    drift (the very failure mode that produced the original README/notebook
    mismatch).

    Returns the validated feature list; raises ``AssertionError`` under
    ``strict`` if there is any mismatch.
    """
    derived = [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS and not is_score_column(c)
    ]
    canonical = set(CANONICAL_FEATURES)
    found = set(derived)

    missing = canonical - found            # in canonical list but not in df
    unexpected = found - canonical         # in df but not in canonical list
    if strict:
        assert not missing and not unexpected, (
            "feature schema drift vs CANONICAL_FEATURES:\n"
            f"  missing from data:   {sorted(missing)}\n"
            f"  unexpected in data:  {sorted(unexpected)}"
        )
    # Cluster sanity: every proxy-cluster member must be a real feature.
    assert set(PROXY_CLUSTER) <= canonical, (
        f"PROXY_CLUSTER not a subset of CANONICAL_FEATURES: "
        f"{sorted(set(PROXY_CLUSTER) - canonical)}"
    )
    return feature_columns(df)


# Self-check on import: the curated lists must be internally consistent.
assert len(AGE_ORDER) == 5 and len(set(AGE_ORDER)) == 5
assert len(CANONICAL_FEATURES) == 51, len(CANONICAL_FEATURES)
assert len(CANONICAL_FEATURES) == len(set(CANONICAL_FEATURES)), "duplicate feature names"
assert set(PROXY_CLUSTER) <= set(CANONICAL_FEATURES)
assert set(EXPERIENCE_FEATURES) <= set(CANONICAL_FEATURES)
