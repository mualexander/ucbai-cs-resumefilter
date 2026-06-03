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

Concurrency (this revision):
- Requests are issued concurrently via asyncio + AsyncOpenAI, bounded by a semaphore.
- Per-call exponential backoff with jitter on 429/5xx/timeout/connection errors,
  honoring a `Retry-After` header when present.
- Every per-row input (style, seed, name, URLs) is a pure function of candidate_id, so
  concurrency NEVER changes which prompt/seed lands on which row -- output is identical
  to the serial version, only faster. The age-language retry, checkpointing, and
  resume-on-restart behavior are preserved exactly.
- The binding rate limit for long completions is usually tokens-per-minute (TPM), not
  requests-per-minute. Size --max-concurrency off your org/project TPM (platform.openai
  .com -> Settings -> Limits); multiple processes share one pool and do not help.

Example:
  python src/llm_resume_generation.py \
    --input data/experiments/resume_text_input_age_balanced_10k.parquet \
    --output data/experiments/synthetic_resumes_fulltext_without_direct_age_proxies.parquet \
    --mode without_direct_age_proxies --max-concurrency 32

Environment:
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
import openai as _openai

# Transient errors we retry with backoff. Deliberately excludes the broad APIError
# base (so 4xx like BadRequest fail fast and are recorded, not retried 6x).
_RETRYABLE_ERRORS = tuple(
    getattr(_openai, name)
    for name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError")
    if hasattr(_openai, name)
) or (Exception,)


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_PROMPT_VERSION = "v6_varied_faithful_plaintext_nolabels"


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
- Never print structured field names, labels, or bucket codes verbatim (e.g., "Company Size: 1000+", "University Tier: 5", "School Tier", "GPA Bucket"). Express such facts in natural language (e.g., "at a large organization") or omit them; never print a school tier at all.
- Do NOT describe the candidate with evaluative age/experience adjectives such as "seasoned", "veteran", "young", "youthful", "mature", "recent graduate", or "fresh graduate". State experience only as the provided facts (e.g., "X years of experience").

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


def write_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp sibling then os.replace, so an interrupted checkpoint never
    leaves a half-written output file. Dispatches on the FINAL path's extension
    (the temp file's `.tmp` suffix must not drive format selection)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    s = path.suffix.lower()
    if s == ".csv":
        df.to_csv(tmp, index=False)
    elif s in {".parquet", ".pq"}:
        df.to_parquet(tmp, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    os.replace(tmp, path)


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
        s = line.strip()
        if s and any(p.search(s) for p in PLACEHOLDER_PATTERNS):
            continue
        if re.fullmatch(r"[-*_=]{3,}", s):   # markdown horizontal rules / divider lines
            continue
        out.append(line)
    cleaned = "\n".join(out)
    # Strip markdown emphasis/headers so output is true plain text (ATS-style),
    # consistent with the external plain-text corpus used in notebook 09.
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)            # **bold**
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)                # __bold__
    cleaned = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", cleaned) # *italic*
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

    education = {c: get(c) for c in ["highest_degree","graduation_year","gpa_bucket"] if get(c) is not None}
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
- Do not print structured field labels or bucket codes (e.g. "Company Size: 1000+", "University Tier: 5"); use natural language or omit.
- Do not use evaluative age/experience adjectives (e.g. "seasoned", "veteran", "young", "mature", "recent graduate").
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
    r"\b(young|youthful|recent graduate|fresh graduate|seasoned|veteran|"
    r"mature professional|mature engineer|elderly|"
    r"older (?:worker|professional|candidate|individual|engineer)|"
    r"decades of (?:experience|industry)|long career|late[- ]career|"   # <- added
    r"junior in age|age \d{2})\b", re.IGNORECASE)
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
# Async generation (one API call, with backoff; semaphore-bounded)
# ---------------------------------------------------------------------

def _retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort read of a Retry-After header off an OpenAI error's response."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


async def acall_chat_completion(
    client: AsyncOpenAI, sem: asyncio.Semaphore, *, model: str, temperature: float, seed: int,
    messages: list[dict], max_retries: int, base_delay: float = 2.0,
) -> str:
    """Single chat completion with exponential backoff + jitter. The semaphore is
    acquired per-attempt and RELEASED during backoff sleeps, so a waiting request
    never burns a concurrency slot. Empty content is treated as retryable."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model, temperature=temperature, seed=seed, messages=messages,
                )
                content = resp.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                last_error = ValueError("Model returned empty content.")
            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                retry_after = _retry_after_seconds(exc)
            else:
                retry_after = None  # empty content -> plain backoff
        # slot released here before sleeping (retry_after is always bound by the
        # try/except/else above)
        if attempt < max_retries:
            delay = retry_after if retry_after is not None else base_delay * (2 ** attempt)
            delay += random.uniform(0.0, 0.5 * delay)  # jitter
            await asyncio.sleep(delay)
    raise RuntimeError(f"Failed after {max_retries + 1} attempts.") from last_error


async def process_row(
    idx: Any, row: pd.Series, *, client: AsyncOpenAI, sem: asyncio.Semaphore,
    mode: str, model: str, temperature: float, base_seed: int, id_column: str, max_retries: int,
) -> dict[str, Any]:
    """Build the prompt deterministically from candidate_id, generate, audit, and
    retry up to 3x ONLY on evaluative age language (attempt 0 keeps the deterministic
    seed so clean rows are unchanged). Mirrors the serial loop exactly."""
    candidate_id = row[id_column]
    style = choose_style(candidate_id)
    seed = (base_seed + stable_int(candidate_id)) % (2**31 - 1)
    target_words = length_target_words(row)
    record = compact_record_for_prompt(row, mode, id_column)
    user_prompt = build_user_prompt(record, mode, style, target_words)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]

    text, issues, used_seed = None, ["age_language"], seed
    for attempt in range(3):
        used_seed = (seed + attempt * 104729) % (2**31 - 1)
        raw = await acall_chat_completion(
            client, sem, model=model, temperature=temperature, seed=used_seed,
            messages=messages, max_retries=max_retries,
        )
        text = clean_generated_resume_text(raw)
        issues = audit_faithfulness(text, record, mode, target_words)
        if "age_language" not in issues:
            break

    return {
        "idx": idx, "generation_mode": mode, "resume_text": text, "llm_model": model,
        "temperature": temperature, "seed": used_seed, "style": style,
        "prompt_version": DEFAULT_PROMPT_VERSION, "generated_at": now_utc_iso(),
        "generation_error": None, "faithfulness_ok": (len(issues) == 0),
        "faithfulness_issues": "; ".join(issues) if issues else None,
    }


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

OUT_COLS = ["generation_mode","resume_text","llm_model","temperature","seed","style",
            "prompt_version","generated_at","generation_error","faithfulness_ok","faithfulness_issues"]


def _prepare_output_df(input_df: pd.DataFrame, output_path: Path, mode: str,
                       id_column: str, overwrite: bool) -> pd.DataFrame:
    """Resume-on-restart: merge any prior results for this mode, then ensure out cols."""
    if output_path.exists() and not overwrite:
        prev = read_dataframe(output_path)
        if {id_column, "generation_mode", "resume_text"}.issubset(prev.columns):
            keep = [c for c in [id_column] + OUT_COLS if c in prev.columns]
            prior = prev[prev["generation_mode"].astype(str) == mode][keep]
            output_df = input_df.merge(prior, on=id_column, how="left")
        else:
            output_df = input_df.copy()
    else:
        output_df = input_df.copy()

    for col in OUT_COLS:
        if col not in output_df.columns:
            output_df[col] = pd.Series([pd.NA] * len(output_df), dtype="object")
    return output_df


def _needs_generation(row: pd.Series, overwrite: bool) -> bool:
    if overwrite:
        return True
    existing = row.get("resume_text")
    return not (isinstance(existing, str) and existing.strip())


async def _run_async(
    output_df: pd.DataFrame, todo_idxs: list, output_path: Path, *, client: AsyncOpenAI,
    mode: str, model: str, temperature: float, base_seed: int, id_column: str,
    max_concurrency: int, max_retries: int, save_every: int,
) -> None:
    sem = asyncio.Semaphore(max_concurrency)
    total = len(todo_idxs)
    state = {"done": 0}  # single-threaded asyncio: mutation between awaits is atomic

    async def worker(idx) -> None:
        row = output_df.loc[idx]
        candidate_id = row[id_column]
        try:
            res = await process_row(
                idx, row, client=client, sem=sem, mode=mode, model=model,
                temperature=temperature, base_seed=base_seed, id_column=id_column,
                max_retries=max_retries,
            )
            for col in OUT_COLS:
                output_df.at[idx, col] = res[col]
            tag = "ok" if res["faithfulness_ok"] else f"issues=[{res['faithfulness_issues']}]"
        except Exception as exc:  # record, never crash the whole run (matches serial)
            output_df.at[idx, "generation_mode"] = mode
            output_df.at[idx, "generation_error"] = str(exc)
            tag = f"ERROR: {exc}"

        state["done"] += 1
        done = state["done"]
        print(f"[{done}/{total}] {id_column}={candidate_id} | {tag}")
        if done % save_every == 0:
            # Blocking write with no await inside -> atomic w.r.t. the event loop.
            write_dataframe_atomic(output_df, output_path)
            print(f"  checkpoint -> {output_path} ({done}/{total})")

    await asyncio.gather(*(worker(i) for i in todo_idxs))


def generate_fulltext_resumes(
    input_path: Path, output_path: Path, mode: str, model: str, api_key: str | None,
    id_column: str, limit: int | None, save_every: int, overwrite: bool,
    temperature: float, base_seed: int, max_concurrency: int = 24, max_retries: int = 6,
    client: AsyncOpenAI | None = None,
) -> pd.DataFrame:
    """Concurrent generation. `client` is injectable for testing; otherwise an
    AsyncOpenAI client is created (with SDK retries disabled — we own backoff)."""
    mode_excludes(mode)  # validate
    input_df = read_dataframe(input_path)
    if limit is not None:
        input_df = input_df.head(limit).copy()
    if id_column not in input_df.columns:
        print(f"Warning: id_column {id_column!r} not found. Creating from row index.")
        input_df[id_column] = [f"row_{i:06d}" for i in range(len(input_df))]

    output_df = _prepare_output_df(input_df, output_path, mode, id_column, overwrite)
    todo_idxs = [idx for idx, row in output_df.iterrows() if _needs_generation(row, overwrite)]
    print(f"Rows: {len(output_df)} total | {len(todo_idxs)} to generate "
          f"| {len(output_df) - len(todo_idxs)} already done | concurrency={max_concurrency}")

    if todo_idxs:
        owns_client = client is None
        if owns_client:
            client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), max_retries=0)

        async def _main():
            try:
                await _run_async(
                    output_df, todo_idxs, output_path, client=client, mode=mode, model=model,
                    temperature=temperature, base_seed=base_seed, id_column=id_column,
                    max_concurrency=max_concurrency, max_retries=max_retries, save_every=save_every,
                )
            finally:
                if owns_client:
                    await client.close()

        asyncio.run(_main())

    write_dataframe_atomic(output_df, output_path)
    ok = output_df["faithfulness_ok"]
    print(f"Done. Wrote {output_path}")
    try:
        print(f"Faithfulness: {int(ok.sum())}/{int(ok.notna().sum())} clean "
              f"({ok.mean():.1%} of generated rows).")
    except Exception:
        pass
    return output_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate full-text resumes from structured synthetic records (concurrent).")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", default="without_direct_age_proxies", choices=sorted(MODE_EXCLUDE_COLUMNS.keys()))
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Default: {DEFAULT_MODEL}")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--base-seed", type=int, default=42, help="Combined with a per-row hash for a deterministic seed.")
    p.add_argument("--api-key", default=None, help="Prefer OPENAI_API_KEY env var.")
    p.add_argument("--id-column", default="candidate_id")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-concurrency", type=int, default=24,
                   help="Max in-flight requests. Size off your org/project TPM, not RPM.")
    p.add_argument("--max-retries", type=int, default=6, help="Per-call backoff retries on 429/5xx/timeout.")
    p.add_argument("--save-every", type=int, default=50, help="Checkpoint after this many completed rows.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    generate_fulltext_resumes(
        input_path=Path(a.input), output_path=Path(a.output), mode=a.mode, model=a.model,
        api_key=a.api_key, id_column=a.id_column, limit=a.limit, save_every=a.save_every,
        overwrite=a.overwrite, temperature=a.temperature, base_seed=a.base_seed,
        max_concurrency=a.max_concurrency, max_retries=a.max_retries,
    )


if __name__ == "__main__":
    main()
