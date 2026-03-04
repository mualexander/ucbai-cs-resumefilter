import pandas as pd

RESUME_SCHEMA_DTYPES = {
    # identifiers / context
    "candidate_id": "string",
    "application_year": "int16",
    "target_role_family": "category",
    "target_role_level": "category",
    "region": "category",

    # education
    "highest_degree": "category",
    "graduation_year": "int16",      # experiment knob (drop in one condition)
    "school_tier": "int8",           # 1..5 (ordinal)
    "gpa_bucket": "category",        # often missing/unknown

    # work history summary
    "years_experience_total": "float32",
    "years_experience_relevant": "float32",
    "num_employers": "int16",
    "avg_tenure_years": "float32",
    "months_since_last_role": "int16",
    "num_gaps_over_6mo": "int8",
    "most_recent_title": "category",
    "most_recent_company_size": "category",
    "management_years": "float32",
    "reports_max": "int16",

    # skills (counts)
    "num_skills_listed": "int16",
    "num_programming_languages": "int8",
    "num_cloud_platforms": "int8",
    "num_databases": "int8",

    # skills (binary flags)
    "skill_python": "boolean",
    "skill_java": "boolean",
    "skill_javascript": "boolean",
    "skill_go": "boolean",
    "skill_kubernetes": "boolean",
    "skill_aws": "boolean",
    "skill_gcp": "boolean",
    "skill_azure": "boolean",
    "skill_sql": "boolean",
    "skill_spark": "boolean",
    "skill_terraform": "boolean",
    "skill_linux": "boolean",
    "skill_ml": "boolean",

    # legacy/modern signals
    "legacy_tech_count": "int8",
    "modern_tech_count": "int8",

    # credentials / signals
    "cert_count": "int8",
    "has_top_cloud_cert": "boolean",
    "github_url_present": "boolean",
    "portfolio_url_present": "boolean",
    "open_source_mentions": "boolean",
    "patent_count": "int16",

    # resume quality proxies
    "resume_word_count": "int16",
    "bullet_count": "int16",
    "quantified_impact_count": "int8",
    "keyword_match_score": "float32",   # 0..1
    "format_clean_score": "float32",    # 0..1

    # comp / preferences
    "salary_expectation_usd": "int32",
    "willing_to_relocate": "boolean",
    "remote_only": "boolean",
}

PROTECTED_SCHEMA_DTYPES = {
    "true_age": "int16",          # optional; can omit and only store age_group
    "age_group": "category",      # <30, 30-39, 40-49, 50-59, 60+
}

LABEL_SCHEMA_DTYPES = {
    "callback": "boolean",
    "interview": "boolean",
    "offer": "boolean",
}

DERIVED_SCHEMA_DTYPES = {
    "estimated_start_year": "int16",      # application_year - years_experience_total
    "tech_recency_score": "float32",      # derived 0..1
    "leadership_signal_score": "float32", # derived
    "stability_score": "float32",         # derived
}

ALL_DTYPES = {}
ALL_DTYPES.update(RESUME_SCHEMA_DTYPES)
ALL_DTYPES.update(PROTECTED_SCHEMA_DTYPES)
ALL_DTYPES.update(LABEL_SCHEMA_DTYPES)
ALL_DTYPES.update(DERIVED_SCHEMA_DTYPES)

df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in ALL_DTYPES.items()})
df.dtypes

CATEGORY_LEVELS = {
    "target_role_family": ["SWE", "SRE", "Data", "PM", "Security", "IT", "Sales"],
    "target_role_level": ["Entry", "Mid", "Senior", "Staff", "Manager", "Director"],

    "region": ["US-West", "US-East", "US-Central", "Canada", "EU", "UK", "India", "Other"],

    "highest_degree": ["None", "HS", "AA", "BS", "MS", "PhD", "Bootcamp"],
    "gpa_bucket": ["unknown", "<3.0", "3.0-3.4", "3.5-3.7", "3.8+"],

    "most_recent_title": [
        "Intern", "Junior Engineer", "Engineer", "Senior Engineer", "Staff Engineer",
        "Principal Engineer", "Engineering Manager", "Senior Engineering Manager",
        "Director", "VP", "Other"
    ],
    "most_recent_company_size": ["1-10", "11-50", "51-200", "201-1000", "1000+"],

    "age_group": ["<30", "30-39", "40-49", "50-59", "60+"],
}

def apply_categories(df: pd.DataFrame) -> pd.DataFrame:
    for col, levels in CATEGORY_LEVELS.items():
        if col in df.columns:
            df[col] = df[col].astype(pd.CategoricalDtype(categories=levels))
    return df

FEATURES_BASE = [
    # context
    "application_year", "target_role_family", "target_role_level", "region",

    # education
    "highest_degree", "school_tier", "gpa_bucket",

    # work history
    "years_experience_total", "years_experience_relevant", "num_employers",
    "avg_tenure_years", "months_since_last_role", "num_gaps_over_6mo",
    "most_recent_title", "most_recent_company_size", "management_years", "reports_max",

    # skills counts
    "num_skills_listed", "num_programming_languages", "num_cloud_platforms", "num_databases",

    # skill flags
    "skill_python", "skill_java", "skill_javascript", "skill_go",
    "skill_kubernetes", "skill_aws", "skill_gcp", "skill_azure",
    "skill_sql", "skill_spark", "skill_terraform", "skill_linux", "skill_ml",

    # legacy/modern
    "legacy_tech_count", "modern_tech_count",

    # credentials/signals
    "cert_count", "has_top_cloud_cert", "github_url_present", "portfolio_url_present",
    "open_source_mentions", "patent_count",

    # resume quality proxies
    "resume_word_count", "bullet_count", "quantified_impact_count",
    "keyword_match_score", "format_clean_score",

    # comp/preferences
    "salary_expectation_usd", "willing_to_relocate", "remote_only",
]

FEATURES_WITH_GRAD_YEAR = FEATURES_BASE + ["graduation_year"]
FEATURES_WITHOUT_GRAD_YEAR = FEATURES_BASE  # knob you asked for


