"""Shared corpus handling for the full-text resume experiments (notebooks 07/08).

Single source of truth for:
- the experiment constants the two notebooks must never disagree on
  (``MODES``, ``FOCAL_MODE``, ``TRUE_PROPORTIONS``, ``SCREEN_RATE``);
- loading the three LLM-generated fulltext parquets, with provenance
  fingerprinting (bless on first run, warn loudly if the corpus changes);
- the faithfulness gate and the independent text-leakage audit (the ``AUDIT``
  regex dict previously existed as two diverging copies).

The fulltext parquets are expensive, irreplaceable LLM artifacts. Everything
here is strictly read-only on them.
"""

from __future__ import annotations

import json

import pandas as pd

from src.paths import EXPERIMENTS_DIR
from src.data_generation import dataset_fingerprint

# ---------------------------------------------------------------------------
# Constants shared by notebooks 07 and 08.
# ---------------------------------------------------------------------------
MODES: list[str] = ["with_proxies", "without_direct_age_proxies", "minimal_proxy"]
FOCAL_MODE: str = "without_direct_age_proxies"   # facially-neutral mode for single-target readouts

# The text corpus is age-BALANCED (~2,500/band); the real population is not.
# Any POOLED ("overall") rate must be reweighted to these true proportions.
# Per-group rates need no reweighting.
TRUE_PROPORTIONS: dict[str, float] = {"<30": 0.25, "30-39": 0.30, "40-49": 0.25, "50-59": 0.15, "60+": 0.05}
SCREEN_RATE: float = 0.45

# Independent text-level leakage audit (re-scans the delivered corpus; the
# generation-time audit ran at write time). Bare ambiguous words ("young",
# "mature") are excluded because they collide with surnames; the name-stripped
# generation audit is authoritative for those.
AUDIT: dict[str, str] = {
    "bracket_placeholder":        r"\[[^\]]+\]",
    "unspecified_or_not_provided": r"\b(?:unspecified|not provided|details not provided)\b",
    "company_school_placeholder": r"\b(?:Company Name|School Name)\b",
    "numbered_certification":     r"\bCertification\s+[0-9]+\b",
    "field_label_artifact":       r"(?:University Tier|School Tier|Company Size|GPA Bucket|Tier:\s*\d|Size:\s*\d)",
    "highest_degree_none":        r"Highest Degree:\s*None",
    "protected_terms":            r"\b(?:true_age|age_group|protected class|fairness)\b",
    "explicit_age_phrase":        r"\b[0-9]{2}\s*(?:years old|year-old)\b",
    "evaluative_age_language":    r"\b(?:youthful|seasoned|veteran|elderly|recent graduate|fresh graduate|"
                                  r"mature professional|mature engineer|decades of experience|long career|late-career)\b",
}


def load_fulltext_corpus(experiments_dir=EXPERIMENTS_DIR, modes=None, check_fingerprints=True):
    """Load the per-mode fulltext parquets and return ``(fulltext, fps)``.

    ``fps`` maps mode -> dataset fingerprint; pass it through to scored-output
    metadata so every derived artifact is traceable to its input corpus.
    On the first run the fingerprints are blessed (written to
    ``fulltext_fingerprints.json``); on later runs a mismatch prints a loud
    warning but does not raise, because an *intentional* regeneration is a
    legitimate state — it just must never be a silent one.
    """
    modes = list(modes or MODES)
    frames, missing = [], []
    for mode in modes:
        p = experiments_dir / f"synthetic_resumes_fulltext_{mode}.parquet"
        if not p.exists():
            missing.append(p)
            continue
        d = pd.read_parquet(p)
        d["generation_mode"] = mode
        frames.append(d)
    if missing:
        raise FileNotFoundError("Missing generated files:\n" + "\n".join(f" - {m}" for m in missing))

    fulltext = pd.concat(frames, ignore_index=True)
    print("loaded:", fulltext.shape)
    print("rows per mode:", fulltext.groupby("generation_mode", observed=True).size().to_dict())
    print("prompt_version:", sorted(fulltext.get("prompt_version", pd.Series(dtype=str)).dropna().unique().tolist()))

    fps = {mode: dataset_fingerprint(f) for mode, f in zip(modes, frames)}
    if check_fingerprints:
        fp_path = experiments_dir / "fulltext_fingerprints.json"
        if fp_path.exists():
            prior = json.loads(fp_path.read_text())
            changed = sorted(m for m in modes if prior.get(m) != fps[m])
            if changed:
                print(f"WARNING: fulltext corpus changed since blessed run: {changed}")
                print("         downstream tables/scored parquets will diverge from earlier results.")
            else:
                print("fulltext fingerprints match blessed corpus")
        else:
            fp_path.write_text(json.dumps(fps, indent=2))
            print("blessed fulltext fingerprints ->", fp_path.name)
    return fulltext, fps


def faithfulness_gate(fulltext):
    """Apply the generation-time faithfulness gate.

    Returns ``(analysis, dropped)`` where ``analysis`` keeps only
    ``faithfulness_ok == True`` rows and ``dropped`` reports the exclusions per
    (mode, age) — on the record that they are small and roughly age-flat, so
    the gate cannot manufacture or hide a gradient.
    """
    ok = fulltext["faithfulness_ok"].fillna(False).astype(bool)
    dropped = (fulltext.assign(dropped=~ok)
               .groupby(["generation_mode", "age_group"], observed=True)["dropped"]
               .agg(dropped_rows="sum", dropped_rate="mean"))
    analysis = fulltext[ok].copy()
    return analysis, dropped


def text_leakage_audit(analysis):
    """Run the independent text audit. Returns ``(audit, clean)``.

    ``audit`` is the per-(mode, pattern) hit table; ``clean`` is a boolean
    Series aligned to ``analysis`` — True where the text trips no pattern.
    Row-level exclusions are applied at aggregation time, not before
    embedding, so the embedding cache (keyed on the full faithful corpus)
    stays valid if the audit patterns evolve.
    """
    rows = []
    for mode, g in analysis.groupby("generation_mode", observed=True):
        t = g["resume_text"].fillna("")
        for label, pat in AUDIT.items():
            m = t.str.contains(pat, case=False, regex=True, na=False)
            rows.append({"generation_mode": mode, "pattern": label,
                         "hits": int(m.sum()), "rate": float(m.mean())})
    audit = pd.DataFrame(rows)

    hard = "|".join(f"(?:{p})" for p in AUDIT.values())
    clean = ~analysis["resume_text"].fillna("").str.contains(hard, case=False, regex=True, na=False)
    return audit, clean
