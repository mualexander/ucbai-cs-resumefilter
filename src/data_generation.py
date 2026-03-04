# src/data_generation.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd


# -----------------------------
# Controlled vocabularies
# -----------------------------

DEFAULT_CATEGORY_LEVELS = {
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


# -----------------------------
# Dtypes schema (pandas-ready)
# -----------------------------

RESUME_SCHEMA_DTYPES = {
    # identifiers / context
    "candidate_id": "string",
    "application_year": "int16",
    "target_role_family": "category",
    "target_role_level": "category",
    "region": "category",

    # education
    "highest_degree": "category",
    "graduation_year": "int16",
    "school_tier": "int8",
    "gpa_bucket": "category",

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
    "keyword_match_score": "float32",  # 0..1
    "format_clean_score": "float32",   # 0..1

    # comp / preferences
    "salary_expectation_usd": "int32",
    "willing_to_relocate": "boolean",
    "remote_only": "boolean",
}

PROTECTED_SCHEMA_DTYPES = {
    "true_age": "int16",
    "age_group": "category",
}

LABEL_SCHEMA_DTYPES = {
    "callback": "boolean",
    "interview": "boolean",
    "offer": "boolean",
}

DERIVED_SCHEMA_DTYPES = {
    "estimated_start_year": "int16",
    "tech_recency_score": "float32",
    "leadership_signal_score": "float32",
    "stability_score": "float32",
}

ALL_DTYPES = {}
ALL_DTYPES.update(RESUME_SCHEMA_DTYPES)
ALL_DTYPES.update(PROTECTED_SCHEMA_DTYPES)
ALL_DTYPES.update(LABEL_SCHEMA_DTYPES)
ALL_DTYPES.update(DERIVED_SCHEMA_DTYPES)


# -----------------------------
# Config + helpers
# -----------------------------

@dataclass(frozen=True)
class GenerationConfig:
    n_samples: int = 10_000
    application_year: int = 2025
    bias_strength: float = 0.10
    seed: int = 42

    # Age group mixture; must sum to 1.0
    p_age_groups: tuple[float, float, float, float, float] = (0.25, 0.30, 0.25, 0.15, 0.05)

    # Quantile threshold for callback (lower => more callbacks)
    callback_quantile: float = 0.60


def apply_categories(df: pd.DataFrame, category_levels: dict | None = None) -> pd.DataFrame:
    levels = category_levels or DEFAULT_CATEGORY_LEVELS
    for col, cats in levels.items():
        if col in df.columns:
            df[col] = df[col].astype(pd.CategoricalDtype(categories=cats))
    return df


def coerce_dtypes(df: pd.DataFrame, dtypes: dict | None = None) -> pd.DataFrame:
    dtypes = dtypes or ALL_DTYPES

    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue

        if dtype == "category":
            df[col] = df[col].astype("category")
        else:
            # For pandas nullable booleans, use "boolean"
            df[col] = df[col].astype(dtype)

    return df


def save_artifacts(
    df: pd.DataFrame,
    out_dir: str | Path,
    config: GenerationConfig,
    parquet_name: str = "synthetic_resumes_full.parquet",
    sample_csv_name: str = "synthetic_resumes_sample.csv",
    metadata_name: str = "generation_metadata.json",
    sample_n: int = 1000,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out / parquet_name, index=False)

    # deterministic sample
    df.sample(min(sample_n, len(df)), random_state=config.seed).to_csv(
        out / sample_csv_name, index=False
    )

    metadata = asdict(config)
    metadata["description"] = "Synthetic resume dataset for age proxy bias experiment"

    with open(out / metadata_name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


# -----------------------------
# Main generator
# -----------------------------

def generate_synthetic_resumes(config: GenerationConfig) -> pd.DataFrame:
    """
    Generate a synthetic resume dataset designed to study proxy-bias leakage.

    - age/age_group is generated for fairness evaluation only (not for model features).
    - graduation_year is included as a configurable strong proxy.
    - callback label includes an adjustable penalty for age >= 40 groups.

    Returns:
        DataFrame with columns matching ALL_DTYPES (where applicable).
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    df = pd.DataFrame()
    df["candidate_id"] = pd.Series([f"cand_{i:06d}" for i in range(n)], dtype="string")
    df["application_year"] = np.int16(config.application_year)

    # --- role context
    df["target_role_family"] = rng.choice(
        DEFAULT_CATEGORY_LEVELS["target_role_family"], size=n, replace=True
    )
    df["target_role_level"] = rng.choice(
        DEFAULT_CATEGORY_LEVELS["target_role_level"], size=n, replace=True
    )
    df["region"] = rng.choice(DEFAULT_CATEGORY_LEVELS["region"], size=n, replace=True)

    # --- age group + true age
    age_groups = rng.choice(
        DEFAULT_CATEGORY_LEVELS["age_group"],
        size=n,
        p=np.array(config.p_age_groups, dtype=float),
        replace=True,
    )
    df["age_group"] = age_groups

    # Sample true age inside each bucket (vectorized via masks)
    true_age = np.empty(n, dtype=np.int16)
    masks = {
        "<30": (age_groups == "<30"),
        "30-39": (age_groups == "30-39"),
        "40-49": (age_groups == "40-49"),
        "50-59": (age_groups == "50-59"),
        "60+": (age_groups == "60+"),
    }
    true_age[masks["<30"]] = rng.integers(22, 30, size=masks["<30"].sum(), endpoint=False).astype(np.int16)
    true_age[masks["30-39"]] = rng.integers(30, 40, size=masks["30-39"].sum(), endpoint=False).astype(np.int16)
    true_age[masks["40-49"]] = rng.integers(40, 50, size=masks["40-49"].sum(), endpoint=False).astype(np.int16)
    true_age[masks["50-59"]] = rng.integers(50, 60, size=masks["50-59"].sum(), endpoint=False).astype(np.int16)
    true_age[masks["60+"]] = rng.integers(60, 67, size=masks["60+"].sum(), endpoint=False).astype(np.int16)

    df["true_age"] = true_age

    # --- education
    df["highest_degree"] = rng.choice(DEFAULT_CATEGORY_LEVELS["highest_degree"], size=n, replace=True)
    df["school_tier"] = rng.integers(1, 6, size=n).astype(np.int8)  # 1..5

    # graduation_year: approx application_year - (age - grad_age)
    grad_age = rng.integers(21, 24, size=n)  # typical undergrad completion age band
    df["graduation_year"] = (config.application_year - true_age + grad_age).astype(np.int16)

    # GPA bucket: often unknown
    df["gpa_bucket"] = rng.choice(
        DEFAULT_CATEGORY_LEVELS["gpa_bucket"], size=n, p=[0.55, 0.10, 0.15, 0.12, 0.08]
    )

    # --- experience: correlated with age
    # start around 22, noisy
    years_exp = (true_age - 22 + rng.normal(0, 2.0, size=n)).clip(0, 45).astype(np.float32)
    df["years_experience_total"] = years_exp
    df["years_experience_relevant"] = (years_exp * rng.uniform(0.70, 1.0, size=n)).astype(np.float32)

    # employers / tenure / gaps
    # more experience => more employers, but not strictly
    df["num_employers"] = np.clip(
        (years_exp / rng.uniform(2.5, 4.5, size=n)) + rng.normal(0, 1.0, size=n),
        0, 25
    ).astype(np.int16)

    # avg tenure loosely correlates with experience and age group
    df["avg_tenure_years"] = np.clip(
        rng.normal(2.5, 1.0, size=n) + (true_age - 30) / 60,
        0.5, 12.0
    ).astype(np.float32)

    df["months_since_last_role"] = np.clip(
        rng.normal(4, 8, size=n), 0, 240
    ).astype(np.int16)

    df["num_gaps_over_6mo"] = np.clip(
        rng.poisson(lam=0.4, size=n) + (df["months_since_last_role"].to_numpy() > 12).astype(int),
        0, 10
    ).astype(np.int8)

    # --- titles correlated with experience (simple mapping)
    def assign_title_vec(y: np.ndarray) -> np.ndarray:
        out = np.empty(y.shape[0], dtype=object)
        out[y < 1.0] = "Intern"
        out[(y >= 1.0) & (y < 3.0)] = "Junior Engineer"
        out[(y >= 3.0) & (y < 7.0)] = "Engineer"
        out[(y >= 7.0) & (y < 12.0)] = "Senior Engineer"
        out[(y >= 12.0) & (y < 18.0)] = "Staff Engineer"
        out[(y >= 18.0) & (y < 25.0)] = "Engineering Manager"
        out[y >= 25.0] = "Director"
        return out

    df["most_recent_title"] = assign_title_vec(years_exp)

    # company size: weak correlation with seniority
    size_levels = DEFAULT_CATEGORY_LEVELS["most_recent_company_size"]
    seniority = np.clip(years_exp / 25.0, 0, 1)
    # Weighted draw: more senior => slightly more likely 1000+
    probs = np.vstack([
        0.15 - 0.05 * seniority,
        0.25 - 0.05 * seniority,
        0.25 + 0.00 * seniority,
        0.20 + 0.03 * seniority,
        0.15 + 0.07 * seniority,
    ]).T
    probs = np.clip(probs, 0.01, None)
    probs = (probs.T / probs.sum(axis=1)).T
    df["most_recent_company_size"] = [rng.choice(size_levels, p=probs[i]) for i in range(n)]

    # management signal
    mgmt_years = np.clip(years_exp - 10.0 + rng.normal(0, 1.5, size=n), 0, 30).astype(np.float32)
    df["management_years"] = mgmt_years
    df["reports_max"] = np.clip(
        (mgmt_years * rng.uniform(2, 8, size=n)) + rng.normal(0, 3, size=n),
        0, 500
    ).astype(np.int16)

    # --- skills counts
    df["num_skills_listed"] = np.clip(rng.normal(22, 10, size=n), 5, 80).astype(np.int16)
    df["num_programming_languages"] = np.clip(rng.normal(4, 2, size=n), 1, 12).astype(np.int8)
    df["num_cloud_platforms"] = np.clip(rng.normal(1.5, 1.0, size=n), 0, 4).astype(np.int8)
    df["num_databases"] = np.clip(rng.normal(2.5, 1.5, size=n), 0, 8).astype(np.int8)

    # --- skill flags (probabilities can depend on role family if you want later)
    def bern(p: float | np.ndarray) -> pd.Series:
        return pd.Series(rng.random(n) < p, dtype="boolean")

    df["skill_python"] = bern(0.70)
    df["skill_java"] = bern(0.45)
    df["skill_javascript"] = bern(0.55)
    df["skill_go"] = bern(0.30)
    df["skill_kubernetes"] = bern(0.55)
    df["skill_aws"] = bern(0.60)
    df["skill_gcp"] = bern(0.30)
    df["skill_azure"] = bern(0.25)
    df["skill_sql"] = bern(0.65)
    df["skill_spark"] = bern(0.25)
    df["skill_terraform"] = bern(0.50)
    df["skill_linux"] = bern(0.75)
    df["skill_ml"] = bern(0.22)

    # --- legacy vs modern tech signals (age leakage)
    modern = np.clip(
        5 - (true_age - 30) / 15 + rng.normal(0, 1.2, size=n),
        0, 10
    ).astype(np.int8)
    legacy = np.clip(
        (true_age - 30) / 10 + rng.normal(0, 1.0, size=n),
        0, 8
    ).astype(np.int8)
    df["modern_tech_count"] = modern
    df["legacy_tech_count"] = legacy

    # --- credentials / signals
    df["cert_count"] = np.clip(rng.poisson(lam=1.2, size=n), 0, 12).astype(np.int8)
    df["has_top_cloud_cert"] = pd.Series(rng.random(n) < 0.10, dtype="boolean")
    df["github_url_present"] = pd.Series(rng.random(n) < 0.55, dtype="boolean")
    df["portfolio_url_present"] = pd.Series(rng.random(n) < 0.25, dtype="boolean")
    df["open_source_mentions"] = pd.Series(rng.random(n) < 0.35, dtype="boolean")
    df["patent_count"] = np.clip(rng.poisson(lam=0.05, size=n), 0, 20).astype(np.int16)

    # --- resume quality proxies
    df["resume_word_count"] = np.clip(rng.normal(650, 180, size=n), 200, 1500).astype(np.int16)
    df["bullet_count"] = np.clip(rng.normal(28, 12, size=n), 5, 80).astype(np.int16)
    df["quantified_impact_count"] = np.clip(rng.poisson(lam=4.0, size=n), 0, 25).astype(np.int8)
    df["keyword_match_score"] = rng.uniform(0.45, 0.98, size=n).astype(np.float32)
    df["format_clean_score"] = rng.uniform(0.60, 1.00, size=n).astype(np.float32)

    # --- compensation / preferences
    salary = 75_000 + (years_exp * 7_000) + (mgmt_years * 3_000) + rng.normal(0, 18_000, size=n)
    df["salary_expectation_usd"] = np.clip(salary, 50_000, 650_000).astype(np.int32)

    df["willing_to_relocate"] = pd.Series(rng.random(n) < 0.35, dtype="boolean")
    df["remote_only"] = pd.Series(rng.random(n) < 0.30, dtype="boolean")

    # --- derived fields
    df["estimated_start_year"] = np.clip(
        (config.application_year - years_exp).round(),
        1960, config.application_year
    ).astype(np.int16)

    # tech recency: more modern tech + recency of employment => higher
    tech_recency = (
        0.6 * (modern.astype(np.float32) / 10.0) +
        0.2 * (1.0 - (df["months_since_last_role"].to_numpy().astype(np.float32) / 240.0)) +
        0.2 * df["keyword_match_score"].to_numpy().astype(np.float32)
    )
    df["tech_recency_score"] = np.clip(tech_recency, 0.0, 1.0).astype(np.float32)

    leadership_signal = (
        0.6 * (mgmt_years / 20.0) +
        0.4 * (df["reports_max"].to_numpy().astype(np.float32) / 200.0)
    )
    df["leadership_signal_score"] = np.clip(leadership_signal, 0.0, 2.0).astype(np.float32)

    stability = (
        0.7 * (df["avg_tenure_years"].to_numpy().astype(np.float32) / 6.0) -
        0.3 * (df["num_gaps_over_6mo"].to_numpy().astype(np.float32) / 5.0)
    )
    df["stability_score"] = np.clip(stability, -1.0, 2.0).astype(np.float32)

    # --- label generation: callback (inject age penalty for 40+)
    merit_score = (
        0.35 * df["keyword_match_score"].to_numpy().astype(np.float32) +
        0.20 * (df["tech_recency_score"].to_numpy().astype(np.float32)) +
        0.20 * (df["years_experience_relevant"].to_numpy().astype(np.float32) / 25.0) +
        0.15 * (df["format_clean_score"].to_numpy().astype(np.float32)) +
        0.10 * (df["quantified_impact_count"].to_numpy().astype(np.float32) / 25.0)
    )

    is_40_plus = np.isin(age_groups, ["40-49", "50-59", "60+"]).astype(np.float32)
    final_score = merit_score - (is_40_plus * np.float32(config.bias_strength))

    threshold = np.quantile(final_score, config.callback_quantile)
    df["callback"] = pd.Series(final_score > threshold, dtype="boolean")

    # placeholders (optional future labels)
    df["interview"] = pd.Series([pd.NA] * n, dtype="boolean")
    df["offer"] = pd.Series([pd.NA] * n, dtype="boolean")

    # Apply categories + dtypes
    df = apply_categories(df)
    df = coerce_dtypes(df)

    return df

