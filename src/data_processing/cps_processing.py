from __future__ import annotations

import numpy as np
import pandas as pd


AGE_BINS = [18, 30, 40, 50, np.inf]
AGE_LABELS = ["<30", "30-39", "40-49", "50+"]

CAREER_STAGE_BINS = [22, 33, 43, 53, np.inf]
CAREER_STAGE_LABELS = ["22-32", "33-42", "43-52", "53+"]

UNEMPLOYED_CODES = [20, 21, 22]
EMPLOYED_CODES = [10, 12]

ANALYSIS_COLUMNS = ["YEAR", "MONTH", "AGE", "EMPSTAT", "DURUNEMP", "WTFINL", "OCC2010"]
DURUNEMP_NIU = 999

LABOR_FORCE_CODES = EMPLOYED_CODES + UNEMPLOYED_CODES   # [10, 12, 20, 21, 22]

def standardize_cps_extract(raw_path, columns=ANALYSIS_COLUMNS, min_year=None):
    """Standardize a raw IPUMS CPS CSV extract into the analysis frame
    (data/interim/cps_clean.parquet).

    - Basic-monthly only: ASEC/supplement rows have no basic-monthly weight
      (WTFINL is NIU), so WTFINL>0 drops them and avoids double-counting March.
    - Recodes DURUNEMP's NIU sentinel (999) to NaN, so a not-in-universe value
      can never inflate a duration mean (and so the downstream dropna works).
    - Keeps only the columns the analysis uses.

    Age/weight analysis filters and the age/career groupings stay in
    clean_cps_base(); this function just defines the basic-monthly universe.
    """
    df = pd.read_csv(raw_path, usecols=lambda c: c in set(columns))
    df = df[df["WTFINL"].notna() & (df["WTFINL"] > 0)]            # basic monthly only
    df["DURUNEMP"] = df["DURUNEMP"].replace(DURUNEMP_NIU, np.nan)  # NIU -> NaN
    if min_year is not None:
        df = df[df["YEAR"] >= min_year]
    return df.reset_index(drop=True)

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add broad age-group buckets."""
    df = df.copy()
    df["age_group"] = pd.cut(
        df["AGE"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    )
    return df


def add_career_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Add career-stage buckets aligned to the project hypothesis."""
    df = df.copy()
    df["career_stage"] = pd.cut(
        df["AGE"],
        bins=CAREER_STAGE_BINS,
        labels=CAREER_STAGE_LABELS,
        right=False,
    )
    return df


def clean_cps_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Base cleaning for CPS person-level analysis.

    Keeps adults with valid person weights.
    Does not filter on employment status or occupation.
    Adds age_group and career_stage.
    """
    df = df.copy()

    df = df[df["AGE"].notna()]
    df = df[df["AGE"] >= 18]

    df = df[df["WTFINL"].notna()]
    df = df[df["WTFINL"] > 0]

    df = add_age_group(df)
    df = add_career_stage(df)

    return df


def build_cps_tech(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build employed tech-occupation subset.

    Notes:
    - Restricts to employed respondents.
    - Restricts to valid OCC2010 codes.
    - Defines tech as OCC2010 1005-1240.
    """
    df = df.copy()

    df = df[df["EMPSTAT"].isin(EMPLOYED_CODES)]
    df = df[df["OCC2010"].notna()]
    df = df[df["OCC2010"] != 9999]

    df["is_tech"] = df["OCC2010"].between(1005, 1240)
    df = df[df["is_tech"]].copy()

    return df


def weighted_mean(
    group: pd.DataFrame,
    value_col: str,
    weight_col: str = "WTFINL",
) -> float:
    """Weighted mean ignoring rows with missing values or invalid weights."""
    valid = (
        group[value_col].notna()
        & group[weight_col].notna()
        & (group[weight_col] > 0)
    )
    if valid.sum() == 0:
        return np.nan

    values = group.loc[valid, value_col]
    weights = group.loc[valid, weight_col]
    return float(np.average(values, weights=weights))


def weighted_rate(
    group: pd.DataFrame,
    indicator_col: str,
    weight_col: str = "WTFINL",
) -> float:
    """Weighted mean of a boolean or 0/1 indicator."""
    valid = (
        group[indicator_col].notna()
        & group[weight_col].notna()
        & (group[weight_col] > 0)
    )
    if valid.sum() == 0:
        return np.nan

    values = group.loc[valid, indicator_col].astype(float)
    weights = group.loc[valid, weight_col]
    return float(np.average(values, weights=weights))

def unemployment_summary_by_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_unemployed"] = df["EMPSTAT"].isin(UNEMPLOYED_CODES)

    lf = df[df["EMPSTAT"].isin(LABOR_FORCE_CODES)]          # denominator = labor force
    unemp_rate = (
        lf.groupby("age_group", observed=True)
        .apply(lambda g: weighted_rate(g, "is_unemployed"))
        .rename("unemployment_rate")
    )

    df_unemp = df[df["is_unemployed"]].copy()
    unemp_duration = (
        df_unemp.groupby("age_group", observed=True)
        .apply(lambda g: weighted_mean(g, "DURUNEMP"))
        .rename("mean_unemployment_duration_weeks")
    )
    return pd.concat([unemp_rate, unemp_duration], axis=1).reset_index()

def unemployment_summary_by_career_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return weighted unemployment rate and weighted mean unemployment duration by career stage.

    Expects a base-cleaned CPS dataframe.
    """
    df = df.copy()
    df["is_unemployed"] = df["EMPSTAT"].isin(UNEMPLOYED_CODES)

    lf = df[df["EMPSTAT"].isin(LABOR_FORCE_CODES)]
    unemp_rate = (
        lf.groupby("career_stage", observed=True)
        .apply(lambda g: weighted_rate(g, "is_unemployed"))
        .rename("unemployment_rate")
    )

    df_unemp = df[df["is_unemployed"]].copy()

    unemp_duration = (
        df_unemp.groupby("career_stage", observed=True)
        .apply(lambda g: weighted_mean(g, "DURUNEMP"))
        .rename("mean_unemployment_duration_weeks")
    )

    return pd.concat([unemp_rate, unemp_duration], axis=1).reset_index()

def unemployment_duration_by_group_over_time(
    df: pd.DataFrame,
    group_col: str,
    year_col: str = "YEAR",
    duration_col: str = "DURUNEMP",
    weight_col: str = "WTFINL",
) -> pd.DataFrame:
    """
    Return weighted unemployment duration by year and a grouping column.

    Year filtering (e.g. complete calendar years only) is the caller's
    responsibility — pass a pre-filtered df. Example group_col values:
    "age_group", "career_stage".
    """
    df = df.copy()
    df["is_unemployed"] = df["EMPSTAT"].isin(UNEMPLOYED_CODES)
    df_unemp = df[df["is_unemployed"]].dropna(subset=[duration_col]).copy()
    return (
        df_unemp.groupby([year_col, group_col], observed=True)
        .apply(lambda g: weighted_mean(g, duration_col, weight_col))
        .rename("mean_duration")
        .reset_index()
    )
