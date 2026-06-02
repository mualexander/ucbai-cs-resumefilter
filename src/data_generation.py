# src/data_generation.py
"""
Synthetic resume generator for the age proxy-bias experiment.

Design notes
------------
The callback label is built from three explicitly separated parts so that the
mechanism producing age disparity is auditable and controllable:

1. base_merit            -- age-NEUTRAL signal (keyword match, formatting,
                            quantified impact). These features are generated
                            independently of age, so base_merit carries no age
                            information by construction.

2. structural_component  -- the "fit to a mid-career target" mechanism: an
                            inverted-U reward for ~mid experience and ~mid
                            management, a reward for modern-tech recency, and a
                            penalty on very-senior titles. Every term here is
                            correlated with age THROUGH legitimate-looking
                            resume features. Scaled by `structural_fit_strength`.

3. explicit_age_penalty  -- a direct, taste-based penalty applied to age groups
                            (the kind of thing fairness work usually imagines).
                            Scaled by `explicit_age_bias`.

final_score = base_merit + structural_component - explicit_age_penalty + noise

Because the two bias channels are independent knobs:
  (structural=1, explicit=1)  -> canonical dataset
  (structural=1, explicit=0)  -> structural-only (the larger, facially-neutral part)
  (structural=0, explicit=1)  -> explicit-only (the smaller, taste-based part)
  (structural=0, explicit=0)  -> NEGATIVE CONTROL: label depends only on
                                 age-neutral merit, so callback disparity ~ 0
                                 even though age remains recoverable from features.

The feature-generation step is unchanged in spirit from the original: age still
drives experience, titles, tech mix, salary, etc. That coupling is the realistic
substrate through which proxies leak, and it is what makes age recoverable
(notebook 05) and what the cluster ablation (notebook 04) removes. The bias knobs
control only how the *label* uses those features, not the features themselves.
"""

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
    "candidate_id": "string",
    "application_year": "int16",
    "target_role_family": "category",
    "target_role_level": "category",
    "region": "category",
    "highest_degree": "category",
    "graduation_year": "int16",
    "school_tier": "int8",
    "gpa_bucket": "category",
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
    "num_skills_listed": "int16",
    "num_programming_languages": "int8",
    "num_cloud_platforms": "int8",
    "num_databases": "int8",
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
    "legacy_tech_count": "int8",
    "modern_tech_count": "int8",
    "cert_count": "int8",
    "has_top_cloud_cert": "boolean",
    "github_url_present": "boolean",
    "portfolio_url_present": "boolean",
    "open_source_mentions": "boolean",
    "patent_count": "int16",
    "resume_word_count": "int16",
    "bullet_count": "int16",
    "quantified_impact_count": "int8",
    "keyword_match_score": "float32",
    "format_clean_score": "float32",
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

# Score-decomposition columns. These are written to the dataframe for auditing
# the label mechanism. They MUST be excluded from any model feature set (the
# notebooks use explicit allow-lists, so this is automatic) -- they are the label.
SCORE_SCHEMA_DTYPES = {
    "score_base_merit": "float32",
    "score_experience_fit": "float32",
    "score_management_fit": "float32",
    "score_recency_reward": "float32",
    "score_senior_title_penalty": "float32",
    "score_structural_component": "float32",
    "score_explicit_age_penalty": "float32",
    "score_noise": "float32",
    "score_final": "float32",
}
SCORE_COMPONENT_COLUMNS = list(SCORE_SCHEMA_DTYPES.keys())

ALL_DTYPES = {}
ALL_DTYPES.update(RESUME_SCHEMA_DTYPES)
ALL_DTYPES.update(PROTECTED_SCHEMA_DTYPES)
ALL_DTYPES.update(LABEL_SCHEMA_DTYPES)
ALL_DTYPES.update(DERIVED_SCHEMA_DTYPES)
ALL_DTYPES.update(SCORE_SCHEMA_DTYPES)


# -----------------------------
# Config + helpers
# -----------------------------

@dataclass(frozen=True)
class GenerationConfig:
    n_samples: int = 10_000
    application_year: int = 2025
    seed: int = 42
    p_age_groups: tuple[float, float, float, float, float] = (0.25, 0.30, 0.25, 0.15, 0.05)
    callback_quantile: float = 0.50

    # --- bias knobs (independent) ---
    # Scales the facially-neutral "fit to mid-career" mechanism (experience/
    # management inverted-U, modern-tech recency reward, senior-title penalty).
    structural_fit_strength: float = 1.0
    # Scales the direct, taste-based penalty applied to older age groups.
    explicit_age_bias: float = 1.0


# Convenience presets for the four corners of the bias design (used in nb 01).
BIAS_SETTINGS = {
    "canonical":        dict(structural_fit_strength=1.0, explicit_age_bias=1.0),
    "structural_only":  dict(structural_fit_strength=1.0, explicit_age_bias=0.0),
    "explicit_only":    dict(structural_fit_strength=0.0, explicit_age_bias=1.0),
    "negative_control": dict(structural_fit_strength=0.0, explicit_age_bias=0.0),
}


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
    df.sample(min(sample_n, len(df)), random_state=config.seed).to_csv(
        out / sample_csv_name, index=False
    )

    metadata = asdict(config)
    metadata["description"] = "Synthetic resume dataset for age proxy bias experiment"
    metadata["score_component_columns"] = SCORE_COMPONENT_COLUMNS
    with open(out / metadata_name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


# -----------------------------
# Feature generation (age is the realistic substrate; unchanged in spirit)
# -----------------------------

def generate_features(config: GenerationConfig) -> pd.DataFrame:
    """Generate the resume feature matrix + protected attrs + derived signals.

    No label is produced here. Age drives experience/title/tech/salary exactly
    as before; this coupling is what makes age recoverable and is independent of
    the bias knobs (which act only on the label).
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    df = pd.DataFrame()
    df["candidate_id"] = pd.Series([f"cand_{i:06d}" for i in range(n)], dtype="string")
    df["application_year"] = np.int16(config.application_year)

    df["target_role_family"] = rng.choice(DEFAULT_CATEGORY_LEVELS["target_role_family"], size=n, replace=True)
    df["target_role_level"] = rng.choice(DEFAULT_CATEGORY_LEVELS["target_role_level"], size=n, replace=True)
    df["region"] = rng.choice(DEFAULT_CATEGORY_LEVELS["region"], size=n, replace=True)

    age_groups = rng.choice(
        DEFAULT_CATEGORY_LEVELS["age_group"], size=n,
        p=np.array(config.p_age_groups, dtype=float), replace=True,
    )
    df["age_group"] = age_groups

    true_age = np.empty(n, dtype=np.int16)
    masks = {g: (age_groups == g) for g in DEFAULT_CATEGORY_LEVELS["age_group"]}
    true_age[masks["<30"]] = rng.integers(22, 30, size=masks["<30"].sum(), endpoint=False)
    true_age[masks["30-39"]] = rng.integers(30, 40, size=masks["30-39"].sum(), endpoint=False)
    true_age[masks["40-49"]] = rng.integers(40, 50, size=masks["40-49"].sum(), endpoint=False)
    true_age[masks["50-59"]] = rng.integers(50, 60, size=masks["50-59"].sum(), endpoint=False)
    true_age[masks["60+"]] = rng.integers(60, 67, size=masks["60+"].sum(), endpoint=False)
    df["true_age"] = true_age

    df["highest_degree"] = rng.choice(DEFAULT_CATEGORY_LEVELS["highest_degree"], size=n, replace=True)
    df["school_tier"] = rng.integers(1, 6, size=n).astype(np.int8)

    # graduation_year: clamp so a candidate cannot "graduate" after applying.
    grad_age = rng.integers(21, 24, size=n)
    grad_year = config.application_year - true_age + grad_age
    grad_year = np.minimum(grad_year, config.application_year)  # FIX: no future grads
    df["graduation_year"] = grad_year.astype(np.int16)

    df["gpa_bucket"] = rng.choice(
        DEFAULT_CATEGORY_LEVELS["gpa_bucket"], size=n, p=[0.55, 0.10, 0.15, 0.12, 0.08]
    )

    # Career starts ~at graduation (age ~22, with spread for early starters) and accrues
    # ~1 year per calendar year, minus modest cumulative gaps. Cap at age-17, not a flat 30,
    # so older candidates carry realistic 25-40 year careers.
    career_start_age = np.clip(grad_age + rng.integers(-1, 2, size=n), 18, 26)
    career_gap_years = np.clip(rng.gamma(shape=1.5, scale=0.8, size=n), 0.0, 6.0)
    years_exp = np.clip(true_age - career_start_age - career_gap_years, 0.0, true_age - 17.0).astype(np.float32)
    df["years_experience_total"] = years_exp
    df["years_experience_relevant"] = (years_exp * rng.uniform(0.75, 1.0, size=n)).astype(np.float32)

    df["num_employers"] = np.clip(
        (years_exp / rng.uniform(2.5, 4.5, size=n)) + rng.normal(0, 1.0, size=n), 0, 25
    ).astype(np.int16)
    df["avg_tenure_years"] = np.clip(
        rng.normal(2.5, 1.0, size=n) + (true_age - 30) / 60, 0.5, 12.0
    ).astype(np.float32)
    df["months_since_last_role"] = np.clip(rng.normal(4, 8, size=n), 0, 240).astype(np.int16)
    df["num_gaps_over_6mo"] = np.clip(
        rng.poisson(lam=0.4, size=n) + (df["months_since_last_role"].to_numpy() > 12).astype(int), 0, 10
    ).astype(np.int8)

    def assign_title_vec(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.empty(y.shape[0], dtype=object)
        for i, yrs in enumerate(y):
            if yrs < 1:
                out[i] = "Intern"
            elif yrs < 3:
                out[i] = rng.choice(["Junior Engineer", "Engineer"], p=[0.7, 0.3])
            elif yrs < 7:
                out[i] = rng.choice(["Engineer", "Senior Engineer"], p=[0.7, 0.3])
            elif yrs < 12:
                out[i] = rng.choice(["Senior Engineer", "Staff Engineer"], p=[0.7, 0.3])
            elif yrs < 18:
                out[i] = rng.choice(["Staff Engineer", "Engineering Manager", "Principal Engineer"], p=[0.5, 0.35, 0.15])
            elif yrs < 25:
                out[i] = rng.choice(["Principal Engineer", "Staff Engineer", "Engineering Manager", "Senior Engineering Manager"], p=[0.30, 0.15, 0.30, 0.25])
            else:
                out[i] = rng.choice(["Principal Engineer", "Staff Engineer", "Engineering Manager", "Senior Engineering Manager", "Director"], p=[0.26, 0.12, 0.27, 0.22, 0.13])
        return out
    df["most_recent_title"] = assign_title_vec(years_exp, rng)

    size_levels = DEFAULT_CATEGORY_LEVELS["most_recent_company_size"]
    seniority = np.clip(years_exp / 25.0, 0, 1)
    probs = np.vstack([
        0.15 - 0.05 * seniority, 0.25 - 0.05 * seniority, 0.25 + 0.00 * seniority,
        0.20 + 0.03 * seniority, 0.15 + 0.07 * seniority,
    ]).T
    probs = np.clip(probs, 0.01, None)
    probs = (probs.T / probs.sum(axis=1)).T
    df["most_recent_company_size"] = [rng.choice(size_levels, p=probs[i]) for i in range(n)]

    mgmt_years = np.clip(0.35 * np.maximum(years_exp - 6.0, 0.0) + rng.normal(0, 1.5, size=n), 0, 12).astype(np.float32)
    df["management_years"] = mgmt_years
    df["reports_max"] = np.clip((mgmt_years * rng.uniform(2, 8, size=n)) + rng.normal(0, 3, size=n), 0, 500).astype(np.int16)

    df["num_skills_listed"] = np.clip(rng.normal(22, 10, size=n), 5, 80).astype(np.int16)
    df["num_programming_languages"] = np.clip(rng.normal(4, 2, size=n), 1, 12).astype(np.int8)
    df["num_cloud_platforms"] = np.clip(rng.normal(1.5, 1.0, size=n), 0, 4).astype(np.int8)
    df["num_databases"] = np.clip(rng.normal(2.5, 1.5, size=n), 0, 8).astype(np.int8)

    def bern(p):
        return pd.Series(rng.random(n) < p, dtype="boolean")
    for col, p in [("skill_python", 0.70), ("skill_java", 0.45), ("skill_javascript", 0.55),
                   ("skill_go", 0.30), ("skill_kubernetes", 0.55), ("skill_aws", 0.60),
                   ("skill_gcp", 0.30), ("skill_azure", 0.25), ("skill_sql", 0.65),
                   ("skill_spark", 0.25), ("skill_terraform", 0.50), ("skill_linux", 0.75),
                   ("skill_ml", 0.22)]:
        df[col] = bern(p)

    modern = np.clip(5 - (true_age - 30) / 15 + rng.normal(0, 1.2, size=n), 0, 10).astype(np.int8)
    legacy = np.clip((true_age - 30) / 10 + rng.normal(0, 1.0, size=n), 0, 8).astype(np.int8)
    df["modern_tech_count"] = modern
    df["legacy_tech_count"] = legacy

    df["cert_count"] = np.clip(rng.poisson(lam=1.2, size=n), 0, 12).astype(np.int8)
    df["has_top_cloud_cert"] = pd.Series(rng.random(n) < 0.10, dtype="boolean")
    df["github_url_present"] = pd.Series(rng.random(n) < 0.55, dtype="boolean")
    df["portfolio_url_present"] = pd.Series(rng.random(n) < 0.25, dtype="boolean")
    df["open_source_mentions"] = pd.Series(rng.random(n) < 0.35, dtype="boolean")
    df["patent_count"] = np.clip(rng.poisson(lam=0.05, size=n), 0, 20).astype(np.int16)

    # age-NEUTRAL resume-quality signals (drawn independent of age -> base_merit)
    df["resume_word_count"] = np.clip(rng.normal(650, 180, size=n), 200, 1500).astype(np.int16)
    df["bullet_count"] = np.clip(rng.normal(28, 12, size=n), 5, 80).astype(np.int16)
    df["quantified_impact_count"] = np.clip(rng.poisson(lam=4.0, size=n), 0, 25).astype(np.int8)
    df["keyword_match_score"] = rng.uniform(0.45, 0.98, size=n).astype(np.float32)
    df["format_clean_score"] = rng.uniform(0.60, 1.00, size=n).astype(np.float32)

    salary = 75_000 + (years_exp * 7_000) + (mgmt_years * 3_000) + rng.normal(0, 18_000, size=n)
    df["salary_expectation_usd"] = np.clip(salary, 50_000, 650_000).astype(np.int32)
    df["willing_to_relocate"] = pd.Series(rng.random(n) < 0.35, dtype="boolean")
    df["remote_only"] = pd.Series(rng.random(n) < 0.30, dtype="boolean")

    df["estimated_start_year"] = np.clip((config.application_year - years_exp).round(), 1960, config.application_year).astype(np.int16)
    tech_recency = (
        0.6 * (modern.astype(np.float32) / 10.0)
        + 0.2 * (1.0 - (df["months_since_last_role"].to_numpy().astype(np.float32) / 240.0))
        + 0.2 * df["keyword_match_score"].to_numpy().astype(np.float32)
    )
    df["tech_recency_score"] = np.clip(tech_recency, 0.0, 1.0).astype(np.float32)
    leadership = 0.6 * (mgmt_years / 20.0) + 0.4 * (df["reports_max"].to_numpy().astype(np.float32) / 200.0)
    df["leadership_signal_score"] = np.clip(leadership, 0.0, 2.0).astype(np.float32)
    stability = 0.7 * (df["avg_tenure_years"].to_numpy().astype(np.float32) / 6.0) - 0.3 * (df["num_gaps_over_6mo"].to_numpy().astype(np.float32) / 5.0)
    df["stability_score"] = np.clip(stability, -1.0, 2.0).astype(np.float32)

    return df


# -----------------------------
# Label decomposition (single source of truth for the scoring mechanism)
# -----------------------------

# Tunable scoring constants (exposed so they are documented, not buried).
EXP_TARGET, EXP_SIGMA = 12.0, 18.0      # peak mid-career fit at ~12 relevant years (~age 34)
MGMT_TARGET, MGMT_SIGMA = 3.0, 4.0      # peak management-fit at ~3 years
W_EXP_FIT = 0.10
W_MGMT_FIT = 0.05
W_RECENCY = 0.08
SENIOR_TITLE_PENALTY = {"Director": 0.03, "VP": 0.03, "Senior Engineering Manager": 0.02, "Principal Engineer": 0.01}
EXPLICIT_AGE_PENALTY = {"<30": 0.00, "30-39": 0.005, "40-49": 0.015, "50-59": 0.022, "60+": 0.032}
NOISE_SD = 0.03


def compute_label_components(df: pd.DataFrame, config: GenerationConfig, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Return the additive components of the callback score for each row.

    final = base_merit + structural_component - explicit_age_penalty + noise
    where structural_component = structural_fit_strength * (
        W_EXP_FIT*exp_fit + W_MGMT_FIT*mgmt_fit + W_RECENCY*recency_reward - senior_title_penalty
    )
    and   explicit_age_penalty = explicit_age_bias * EXPLICIT_AGE_PENALTY[age_group]

    base_merit uses only age-neutral features, so at (structural=0, explicit=0)
    the score carries no age information and callback disparity vanishes.
    """
    if rng is None:
        # deterministic noise stream, seeded off the config (kept separate from
        # feature rng so noise is identical across bias settings on the same seed)
        rng = np.random.default_rng(config.seed + 1)

    n = len(df)
    keyword = df["keyword_match_score"].to_numpy(dtype=np.float32)
    fmt = df["format_clean_score"].to_numpy(dtype=np.float32)
    quant = df["quantified_impact_count"].to_numpy(dtype=np.float32)
    yr_rel = df["years_experience_relevant"].to_numpy(dtype=np.float32)
    mgmt = df["management_years"].to_numpy(dtype=np.float32)
    modern = df["modern_tech_count"].to_numpy(dtype=np.float32)
    title = df["most_recent_title"].astype(str).to_numpy()
    age_group = df["age_group"].astype(str)

    # 1) age-neutral merit
    base_merit = (0.35 * keyword + 0.20 * fmt + 0.15 * (quant / 25.0) + 0.15).astype(np.float32)

    # 2) structural fit-to-mid-career terms
    exp_fit = np.exp(-((yr_rel - EXP_TARGET) ** 2) / (2.0 * EXP_SIGMA ** 2)).astype(np.float32)
    mgmt_fit = np.exp(-((mgmt - MGMT_TARGET) ** 2) / (2.0 * MGMT_SIGMA ** 2)).astype(np.float32)
    recency_reward = np.clip(modern / 10.0, 0.0, 1.0).astype(np.float32)
    senior_title_penalty = np.array(
        [SENIOR_TITLE_PENALTY.get(t, 0.0) for t in title], dtype=np.float32
    )

    s = np.float32(config.structural_fit_strength)
    structural_component = (
        s * (W_EXP_FIT * exp_fit + W_MGMT_FIT * mgmt_fit + W_RECENCY * recency_reward - senior_title_penalty)
    ).astype(np.float32)

    # 3) explicit, taste-based age penalty
    e = np.float32(config.explicit_age_bias)
    explicit_age_penalty = (e * age_group.map(EXPLICIT_AGE_PENALTY).to_numpy(dtype=np.float32)).astype(np.float32)

    noise = rng.normal(0, NOISE_SD, size=n).astype(np.float32)
    final = (base_merit + structural_component - explicit_age_penalty + noise).astype(np.float32)

    return pd.DataFrame({
        "score_base_merit": base_merit,
        "score_experience_fit": (s * W_EXP_FIT * exp_fit).astype(np.float32),
        "score_management_fit": (s * W_MGMT_FIT * mgmt_fit).astype(np.float32),
        "score_recency_reward": (s * W_RECENCY * recency_reward).astype(np.float32),
        "score_senior_title_penalty": (-s * senior_title_penalty).astype(np.float32),
        "score_structural_component": structural_component,
        "score_explicit_age_penalty": (-explicit_age_penalty).astype(np.float32),
        "score_noise": noise,
        "score_final": final,
    }, index=df.index)


# -----------------------------
# Main generator
# -----------------------------

def generate_synthetic_resumes(config: GenerationConfig, attach_score_components: bool = True) -> pd.DataFrame:
    """Generate features, compute the decomposed callback score, threshold to a label.

    The callback threshold is the `callback_quantile` of the final score, so the
    overall callback rate is fixed at (1 - callback_quantile) regardless of the
    bias knobs -- the knobs reshape WHO gets called back, not how many.
    """
    df = generate_features(config)
    comp = compute_label_components(df, config)

    threshold = float(np.quantile(comp["score_final"].to_numpy(), config.callback_quantile))
    df["callback"] = pd.Series(comp["score_final"].to_numpy() > threshold, index=df.index, dtype="boolean")
    df["interview"] = pd.Series([pd.NA] * len(df), dtype="boolean")
    df["offer"] = pd.Series([pd.NA] * len(df), dtype="boolean")

    if attach_score_components:
        for c in SCORE_COMPONENT_COLUMNS:
            df[c] = comp[c].to_numpy()

    df = apply_categories(df)
    df = coerce_dtypes(df)
    return df
