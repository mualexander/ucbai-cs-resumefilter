"""
Generate full-text software-engineering resumes from structured synthetic resume records.

Reads a CSV/Parquet of structured synthetic resumes, sends each row to an LLM, and
writes a CSV/Parquet with generated resume text plus reproducibility metadata and a
faithfulness audit.

Design:
- Facts are locked: the model may only use facts present in the structured record for
  that row, and may not invent employers, dates, metrics, project names, etc.
- Presentation is varied: a deterministic per-candidate STYLE controls layout, section
  order, and verbosity, so resumes do not share a single house style (which would
  inflate text-similarity baselines).
- Length is age-neutral: the target length is the row's `resume_word_count`, which is
  generated independently of age, so longer resumes do not reintroduce a length-by-age
  confound.
- Every row records model, temperature, seed, style, and prompt version, and is audited
  for fact preservation and leakage.

Example:
  python src/llm_resume_generation.py \
    --input data/baseline/synthetic_resumes_full.parquet \
    --output data/experiments/synthetic_resumes_fulltext_without_direct_age_proxies.csv \
    --mode without_direct_age_proxies --limit 25

Environment:
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_PROMPT_VERSION = "v4_varied_faithful_lengthtargeted"


# ---------------------------------------------------------------------
# Prompt modes
# ---------------------------------------------------------------------

ALWAYS_EXCLUDE_COLUMNS = {
    "true_age", "age", "age_group",
    "callback", "interview", "offer", "selected", "outcome", "target", "label",
    "salary_expectation_usd", "willing_to_relocate", "remote_only",
    # internal composite/quality scores: the model can't act on them and some are
    # label-correlated. Keep them out of resume text entirely.
    "tech_recency_score", "leadership_signal_score", "stability_score",
    "keyword_match_score", "format_clean_score", "bullet_count",
    "quantified_impact_count",
    # this script's own output columns
    "resume_text", "llm_model", "temperature", "seed", "style",
    "prompt_version", "generated_at", "generation_error", "generation_mode",
    "faithfulness_ok", "faithfulness_issues",
}

# `resume_word_count` is used as a length target, not printed as a fact.
LENGTH_TARGET_COLUMN = "resume_word_count"

MODE_EXCLUDE_COLUMNS = {
    "with_proxies": set(),
    "without_direct_age_proxies": {
        "graduation_year", "estimated_start_year",
    },
    "minimal_proxy": {
        "graduation_year", "estimated_start_year",
        "years_experience_total", "years_experience_relevant",
        "num_employers", "avg_tenure_years", "months_since_last_role",
        "num_gaps_over_6mo", "management_years", "reports_max",
        "legacy_tech_count", "modern_tech_count",
    },
}


def mode_excludes(mode: str) -> set[str]:
    if mode not in MODE_EXCLUDE_COLUMNS:
        valid = ", ".join(sorted(MODE_EXCLUDE_COLUMNS))
        raise ValueError(f"Invalid mode {mode!r}. Valid modes: {valid}")
    return ALWAYS_EXCLUDE_COLUMNS | MODE_EXCLUDE_COLUMNS[mode]


def is_excluded(col: str, exclude: set[str]) -> bool:
    # Exclude listed columns AND any label-internal score_* column from the
    # regenerated dataset.
    return col in exclude or col.startswith("score_")


# ---------------------------------------------------------------------
# Style variation (presentation only; never changes facts)
# ---------------------------------------------------------------------

STYLES = [
    "Summary-forward: open with a 3-4 sentence professional summary, then Skills, "
    "Experience, Education. Moderate detail in experience bullets.",
    "Skills-forward: lead with a detailed, grouped Skills section, then a brief summary, "
    "then concise Experience and Education.",
    "Chronological narrative: minimal summary, emphasis on a detailed reverse-chronological "
    "Experience section with fuller responsibility descriptions.",
    "Compact one-page style: terse summary, tight bullet points, dense Skills line.",
    "Expanded style: a fuller summary paragraph and longer responsibility descriptions "
    "per role, Skills grouped by category.",
    "Impact-oriented: experience bullets framed around responsibilities and scope "
    "(without inventing specific numbers), Skills and Education after.",
    "Functional layout: a Skills/Competencies section organized by theme first, then a "
    "shorter chronological Experience section.",
    "Plain professional: straightforward Summary, Skills, Experience, Education ordering "
    "with balanced, readable detail.",
]


def choose_style(candidate_id: Any) -> str:
    return STYLES[stable_int(candidate_id) % len(STYLES)]


# ---------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You convert structured synthetic resume records into realistic plain-text software engineering resumes.

FACTS (strict):
- Use only the facts provided in the structured input. Do not introduce new specific facts.
- Use candidate_identity.full_name as the resume name. Never print candidate_reference.
- Do NOT invent employers, job titles, schools, degrees, dates, graduation years, employment dates,
  certifications, awards, metrics, numbers, percentages, project names, locations, database names,
  client names, or technologies that are not in the input.
- Only include URLs explicitly present in the input; if present, copy them exactly.
- If a specific name/date/number is not provided, omit it. Never use bracketed placeholders or
  phrases like "unspecified", "not provided", "Company Name", "Certification 1".
- If highest_degree is "None", omit the Education section entirely.
- If only a certification count is given, refer to the count generally; do not invent certification names.
- Always include a Skills section that lists, by name, every skill in skills.explicit_skills. Never leave the Skills section empty.
- Do not mention projects unless project details are provided.

NO AGE SIGNAL:
- Never mention age, age group, demographics, protected class, or fairness/label terms.
- Never imply age with words like "young", "recent graduate", "seasoned", "veteran", "late-career",
  "decades of", "long career", or similar.
- Do not output internal field or score names.

ELABORATION (for realism, within the facts):
- You MAY describe responsibilities and scope generically in a way that is consistent with the given
  title and skills, to make the resume read naturally. This elaboration must remain generic and must
  NOT introduce specific employers, dates, numbers, or named projects.

STYLE & LENGTH:
- Follow the requested style/layout and target length as closely as the facts allow.
- Plain text only. Use clear section headers. Keep it professional and ATS-readable.
- Output only the resume text.
""".strip()


# ---------------------------------------------------------------------
# Helpers (names, URLs, IO)
# ---------------------------------------------------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_dataframe(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".csv":
        return pd.read_csv(path)
    if s in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = path.suffix.lower()
    if s == ".csv":
        df.to_csv(path, index=False); return
    if s in {".parquet", ".pq"}:
        df.to_parquet(path, index=False); return
    raise ValueError(f"Unsupported output format: {path.suffix}")


def boolish(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


SKILL_MAP = {
    "skill_python": "Python", "skill_java": "Java", "skill_javascript": "JavaScript",
    "skill_go": "Go", "skill_kubernetes": "Kubernetes", "skill_aws": "AWS",
    "skill_gcp": "GCP", "skill_azure": "Azure", "skill_sql": "SQL",
    "skill_spark": "Spark", "skill_terraform": "Terraform", "skill_linux": "Linux",
    "skill_ml": "machine learning",
}


def skill_list_from_row(row: pd.Series) -> list[str]:
    return [label for col, label in SKILL_MAP.items()
            if col in row.index and boolish(row[col]) is True]


FIRST_NAMES = ["Alex","Jordan","Taylor","Morgan","Casey","Riley","Avery","Quinn","Cameron","Drew",
               "Jamie","Skyler","Reese","Rowan","Emerson","Hayden","Dakota","Finley","Kai","Logan"]
LAST_NAMES = ["Chen","Patel","Garcia","Nguyen","Kim","Brown","Johnson","Lee","Martinez","Singh",
              "Davis","Wilson","Anderson","Thomas","Moore","Clark","Lewis","Walker","Hall","Young"]


def stable_int(value: Any) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def synthetic_person_name(candidate_id: Any) -> dict[str, str]:
    seed = stable_int(candidate_id)
    first = FIRST_NAMES[seed % len(FIRST_NAMES)]
    last = LAST_NAMES[(seed // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return {"first_name": first, "last_name": last, "full_name": f"{first} {last}"}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def synthetic_github_url(candidate_id: Any, first: str, last: str) -> str:
    suffix = 100 + (stable_int(candidate_id) % 900)
    return f"https://github.com/{_slug(first)}{_slug(last)}{suffix}"


def synthetic_portfolio_url(candidate_id: Any, first: str, last: str) -> str:
    suffix = 100 + (stable_int(candidate_id) % 900)
    return f"https://{_slug(first)}{_slug(last)}{suffix}.dev"


# ---------------------------------------------------------------------
# Placeholder cleanup
# ---------------------------------------------------------------------

PLACEHOLDER_PATTERNS = [
    re.compile(r"\[[^\]]+\]"),
    re.compile(r"\bunspecified\b", re.IGNORECASE),
    re.compile(r"\bnot provided\b", re.IGNORECASE),
    re.compile(r"\bdetails not provided\b", re.IGNORECASE),
    re.compile(r"\bCompany Name\b", re.IGNORECASE),
    re.compile(r"\bSchool Name\b", re.IGNORECASE),
    re.compile(r"\bCertification [0-9]+\b", re.IGNORECASE),
    re.compile(r"\bHighest Degree:\s*None\b", re.IGNORECASE),
]


def clean_generated_resume_text(text: str) -> str:
    out = []
    for line in text.splitlines():
        if line.strip() and any(p.search(line.strip()) for p in PLACEHOLDER_PATTERNS):
            continue
        out.append(line)
    cleaned = "\n".join(out)
    # Strip markdown emphasis/headers so output is true plain text (ATS-style),
    # consistent with the external plain-text corpus used in notebook 09.
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)   # **bold**
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)        # __bold__
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)  # # headers
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def find_placeholder_artifacts(text: str) -> list[str]:
    arts = []
    for p in PLACEHOLDER_PATTERNS:
        arts.extend(m.group(0) for m in p.finditer(text))
    return arts


# ---------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------

def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if pd.isna(obj):
        return None
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def compact_record_for_prompt(row: pd.Series, mode: str, id_column: str) -> dict[str, Any]:
    exclude = mode_excludes(mode)

    def has(col): return col in row.index and not is_excluded(col, exclude) and not pd.isna(row[col])
    def get(col, default=None):
        if has(col):
            v = row[col]
            if isinstance(v, str):
                v = v.strip()
                return default if v == "" else v
            return v
        return default

    rec: dict[str, Any] = {}
    candidate_ref = str(row[id_column]) if id_column in row.index else "unknown_candidate"
    rec["candidate_reference"] = candidate_ref
    identity = synthetic_person_name(candidate_ref)
    rec["candidate_identity"] = identity

    target = {c: get(c) for c in ["application_year","target_role_family","target_role_level","region"] if get(c) is not None}
    if target: rec["target_context"] = target

    education = {c: get(c) for c in ["highest_degree","graduation_year","school_tier","gpa_bucket"] if get(c) is not None}
    if education: rec["education"] = education

    experience = {c: get(c) for c in ["years_experience_total","years_experience_relevant","num_employers",
                                       "avg_tenure_years","months_since_last_role","num_gaps_over_6mo",
                                       "most_recent_title","most_recent_company_size","management_years",
                                       "reports_max","estimated_start_year"] if get(c) is not None}
    if experience: rec["experience_summary"] = experience

    skills = skill_list_from_row(row)
    skill_counts = {c: get(c) for c in ["num_skills_listed","num_programming_languages","num_cloud_platforms",
                                        "num_databases","legacy_tech_count","modern_tech_count"] if get(c) is not None}
    if skills or skill_counts:
        rec["skills"] = {"explicit_skills": skills, "skill_counts": skill_counts}

    credentials = {c: get(c) for c in ["cert_count","has_top_cloud_cert","open_source_mentions","patent_count"] if get(c) is not None}
    if boolish(get("github_url_present")):
        credentials["github_url"] = synthetic_github_url(candidate_ref, identity["first_name"], identity["last_name"])
    if boolish(get("portfolio_url_present")):
        credentials["portfolio_url"] = synthetic_portfolio_url(candidate_ref, identity["first_name"], identity["last_name"])
    if credentials: rec["credentials_and_signals"] = credentials

    return make_json_safe(rec)


def length_target_words(row: pd.Series) -> int | None:
    if LENGTH_TARGET_COLUMN in row.index and not pd.isna(row[LENGTH_TARGET_COLUMN]):
        try:
            return int(row[LENGTH_TARGET_COLUMN])
        except Exception:
            return None
    return None


def build_user_prompt(record: dict[str, Any], mode: str, style: str, target_words: int | None) -> str:
    record_json = json.dumps(record, indent=2, ensure_ascii=False)
    length_line = (f"- Aim for approximately {target_words} words (a soft target; prioritize the facts)."
                   if target_words else "- Use a natural resume length for the given facts.")
    return f"""
Create a full-text software engineering resume from the structured synthetic resume record below.

Generation mode: {mode}

Requested style/layout:
{style}

Structured record:
{record_json}

Constraints:
- Preserve the factual content; do not add new specific facts (no invented employers, dates, numbers, or project names).
- Use candidate_identity.full_name as the name; never print candidate_reference.
- Do not mention age, age group, callbacks, labels, fairness, or internal field/score names.
- Do not use bracketed placeholders or "unspecified"/"not provided".
- If highest_degree is "None", omit Education.
- Include GitHub/portfolio URLs only if present, copied exactly.
- Do not invent certification names from a count; mention the count generally.
- List every skill in skills.explicit_skills by name in a Skills section; do not leave it empty.
{length_line}
- Output only the resume text.
""".strip()


# ---------------------------------------------------------------------
# Faithfulness audit
# ---------------------------------------------------------------------

AGE_LANGUAGE = re.compile(
    r"\b(young|recent graduate|seasoned|veteran|late[- ]career|early[- ]career|"
    r"decades of|long career|older|elderly|junior in age|age \d+)\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19[6-9]\d|20[0-2]\d)\b")


def audit_faithfulness(text: str, record: dict[str, Any], mode: str, target_words: int | None) -> list[str]:
    """Return a list of issue strings (empty == clean). Checks fact preservation + leakage."""
    issues: list[str] = []
    low = text.lower()

    identity = record.get("candidate_identity", {})
    if identity.get("full_name") and identity["full_name"].lower() not in low:
        issues.append("name_missing")

    exp = record.get("experience_summary", {})
    title = exp.get("most_recent_title")
    if title and str(title).lower() not in low:
        issues.append("title_missing")

    skills = record.get("skills", {}).get("explicit_skills", [])
    if skills:
        missing = [s for s in skills if s.lower() not in low]
        if len(missing) > max(1, len(skills) // 5):  # allow a small slip
            issues.append(f"skills_missing:{missing}")

    # Leakage: no calendar years in the no-date modes (any year is hallucinated/leaked)
    if mode != "with_proxies":
        yrs = YEAR_PATTERN.findall(text)
        if yrs:
            issues.append(f"unexpected_years:{sorted(set(yrs))[:5]}")

    # Strip the candidate's own name before scanning, so a surname like "Young"
    # is not mistaken for age language; body phrases like "young professional" still flag.
    scan_text = text
    full_name = identity.get("full_name", "")
    if full_name:
        scan_text = re.sub(re.escape(full_name), " ", scan_text, flags=re.IGNORECASE)
        for part in full_name.split():
            scan_text = re.sub(rf"\b{re.escape(part)}\b", " ", scan_text, flags=re.IGNORECASE)
    if AGE_LANGUAGE.search(scan_text):
        issues.append("age_language")

    arts = find_placeholder_artifacts(text)
    if arts:
        issues.append(f"placeholder_artifacts:{arts[:5]}")

    wc = len(text.split())
    if wc < 60:
        issues.append(f"too_short:{wc}")
    if target_words and wc > 3 * target_words:
        issues.append(f"too_long:{wc}_vs_target_{target_words}")

    return issues


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def generate_resume_text(client: OpenAI, record, mode, style, target_words, model, temperature,
                         seed: int, max_retries: int = 3, retry_sleep_seconds: float = 3.0) -> str:
    user_prompt = build_user_prompt(record, mode, style, target_words)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, temperature=temperature, seed=seed,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_prompt}],
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Model returned empty content.")
            return content.strip()
        except Exception as exc:
            last_error = exc
            print(f"Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds * attempt)
    raise RuntimeError(f"Failed after {max_retries} attempts.") from last_error


def generate_fulltext_resumes(input_path, output_path, mode, model, api_key, id_column,
                              limit, sleep_seconds, save_every, overwrite, temperature, base_seed) -> pd.DataFrame:
    mode_excludes(mode)  # validate
    input_df = read_dataframe(input_path)
    if limit is not None:
        input_df = input_df.head(limit).copy()
    if id_column not in input_df.columns:
        print(f"Warning: id_column {id_column!r} not found. Creating from row index.")
        input_df[id_column] = [f"row_{i:06d}" for i in range(len(input_df))]

    out_cols = ["generation_mode","resume_text","llm_model","temperature","seed","style",
                "prompt_version","generated_at","generation_error","faithfulness_ok","faithfulness_issues"]

    if output_path.exists() and not overwrite:
        prev = read_dataframe(output_path)
        if {id_column, "generation_mode", "resume_text"}.issubset(prev.columns):
            keep = [c for c in [id_column] + out_cols if c in prev.columns]
            prior = prev[prev["generation_mode"].astype(str) == mode][keep]
            output_df = input_df.merge(prior, on=id_column, how="left")
        else:
            output_df = input_df.copy()
    else:
        output_df = input_df.copy()

    for col in out_cols:
        if col not in output_df.columns:
            output_df[col] = pd.Series([pd.NA] * len(output_df), dtype="object")

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    total = len(output_df)

    for idx, row in output_df.iterrows():
        existing = row.get("resume_text")
        if not overwrite and isinstance(existing, str) and existing.strip():
            continue
        candidate_id = row[id_column]
        style = choose_style(candidate_id)
        seed = (base_seed + stable_int(candidate_id)) % (2**31 - 1)
        target_words = length_target_words(row)
        print(f"Generating {idx + 1}/{total} | {id_column}={candidate_id} | mode={mode}")
        try:
            record = compact_record_for_prompt(row, mode, id_column)
            text = generate_resume_text(client, record, mode, style, target_words, model, temperature, seed)
            text = clean_generated_resume_text(text)
            issues = audit_faithfulness(text, record, mode, target_words)
            output_df.at[idx, "generation_mode"] = mode
            output_df.at[idx, "resume_text"] = text
            output_df.at[idx, "llm_model"] = model
            output_df.at[idx, "temperature"] = temperature
            output_df.at[idx, "seed"] = seed
            output_df.at[idx, "style"] = style
            output_df.at[idx, "prompt_version"] = DEFAULT_PROMPT_VERSION
            output_df.at[idx, "generated_at"] = now_utc_iso()
            output_df.at[idx, "generation_error"] = None
            output_df.at[idx, "faithfulness_ok"] = (len(issues) == 0)
            output_df.at[idx, "faithfulness_issues"] = "; ".join(issues) if issues else None
        except Exception as exc:
            output_df.at[idx, "generation_mode"] = mode
            output_df.at[idx, "generation_error"] = str(exc)
            print(f"Failed {id_column}={candidate_id}: {exc}")
        if (idx + 1) % save_every == 0:
            write_dataframe(output_df, output_path)
            print(f"Saved checkpoint to {output_path}")
        time.sleep(sleep_seconds)

    write_dataframe(output_df, output_path)
    ok = output_df["faithfulness_ok"]
    print(f"Done. Wrote {output_path}")
    try:
        print(f"Faithfulness: {int(ok.sum())}/{int(ok.notna().sum())} clean "
              f"({ok.mean():.1%} of generated rows).")
    except Exception:
        pass
    return output_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate full-text resumes from structured synthetic records.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", default="without_direct_age_proxies", choices=sorted(MODE_EXCLUDE_COLUMNS.keys()))
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Default: {DEFAULT_MODEL}")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--base-seed", type=int, default=42, help="Combined with a per-row hash for a deterministic seed.")
    p.add_argument("--api-key", default=None, help="Prefer OPENAI_API_KEY env var.")
    p.add_argument("--id-column", default="candidate_id")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep-seconds", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    generate_fulltext_resumes(
        input_path=Path(a.input), output_path=Path(a.output), mode=a.mode, model=a.model,
        api_key=a.api_key, id_column=a.id_column, limit=a.limit, sleep_seconds=a.sleep_seconds,
        save_every=a.save_every, overwrite=a.overwrite, temperature=a.temperature, base_seed=a.base_seed,
    )


if __name__ == "__main__":
    main()
