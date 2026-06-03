# Résumé Screening Proxy Bias Experiment

## Executive Summary

This project asks whether résumé-screening systems can produce **age-related disparities without using age as an input** — and, when they do, what actually drives the direction and size of the gap.

The work combines a controlled synthetic testbed, U.S. labor-market context (CPS), full-text résumé screening against job descriptions and against incumbent profiles, and a supplementary real-résumé validation. Across these settings the finding is consistent and specific:

> Facially-neutral screening **transmits** age disparities rather than amplifying them. Job-relevant features (experience, tenure, title, technology mix, salary) act as a redundant proxy for age, so removing any single proxy does nothing. What sets the **direction** of the disparity is not the résumés but the **screening target's composition**: scoring the same résumés against a mid-career incumbent pool penalizes older candidates, while scoring them against a senior job description favors them.

The qualitative results are robust; the one place the controlled mechanism does **not** cleanly reproduce on real data — the incumbent-similarity swap in Notebook 09 — is reported as inconclusive rather than smoothed over.

### Headline: where the score gap comes from

Decomposing the structured callback-score gap between the strongest mid-career band (`30-39`) and the oldest band (`60+`) — a raw gap of **0.142** — attributes it almost entirely to feature structure, not to any explicit age term:

| Source | Share of the `30-39` vs `60+` score gap |
|---|---:|
| **Structural** (job-relevant features that correlate with age) | **82%** |
| **Explicit** age term in the label generator | 19% |
| **Base merit** (age-independent quality) | −1% |

The explicit age knob contributes a minority of the gap; the dominant channel is the correlated feature structure that survives every "remove the obvious proxy" fix tested below. (Shares are rounded and need not sum to exactly 100%.)

---

## Research Question

Can supervised and embedding-based résumé-screening systems reproduce age-related disparities without using explicit age inputs — and what determines the direction and magnitude of those disparities?

More specifically:

- Do structured résumé features predict callbacks in ways that disadvantage older candidates, and does the model *amplify* the disparity in its labels or merely *transmit* it?
- Does removing `graduation_year` — or any single proxy — materially reduce the disparity?
- Can age group still be inferred once age and its obvious proxies are removed, and does mere predictability imply a disparity?
- Do similarity-based screens favor whoever resembles the reference population, and does swapping the reference reverse the gradient?
- Does full-text embedding behavior depend on whether candidates are scored against a job description or against incumbent profiles?
- Do the synthetic mechanisms appear on real résumé text?

---

## Project Structure

| Notebook | Purpose |
|---|---|
| `00_cps_data_prep.ipynb` | Real-world labor-market context from CPS unemployment-duration data. |
| `01_data_generation.ipynb` | Generates the controlled synthetic résumé dataset (50,000 rows) with known age groups, callback labels, and a built-in negative control; decomposes the score gap. |
| `02_model_training.ipynb` | Trains the baseline supervised screening models (logistic regression, gradient boosting). |
| `03_fairness_evaluation.ipynb` | Selection rates, true-positive rates, disparate impact, equal-opportunity difference by age; negative-control check. |
| `04_ablation_graduation_year.ipynb` | Tests removing `graduation_year` and the full proxy cluster; incremental ablation with confidence intervals. |
| `05_age_group_predictability.ipynb` | Recovers age group from résumé features with age excluded; recoverability-vs-disparity identity. |
| `06_similarity_screening_experiment.ipynb` | Structured similarity screening; reference-composition swap. |
| `07_nlp_fulltext_resume_experiment.ipynb` | Full-text résumés scored against a job-description seniority ladder. |
| `08_fulltext_incumbent_similarity_experiment.ipynb` | Full-text incumbent-profile similarity; reference swap with embedding-anisotropy correction. |
| `09_external_resume_career_stage_validation.ipynb` | Applies the framework to real Kaggle résumés with inferred career-stage proxies. |

A single canonical feature list (`src/features.py`, 51 features) is the source of truth shared by every structured notebook; each one calls `validate_features(df)`, which fails loudly on any schema drift. Proportion comparisons throughout use Wilson confidence intervals.

---

## Real-World Labor Market Context

To ground the project, U.S. labor-market data was analyzed using the Current Population Survey (CPS) via IPUMS.

### Methodology

- Data source: IPUMS CPS Basic Monthly
- Population: individuals aged 18+
- Weighting: CPS person weights (`WTFINL`)
- Unemployment definition: `EMPSTAT ∈ {20, 21, 22}`
- Unemployment duration: `DURUNEMP` (weeks)
- Trend: 3-year rolling average

### Key CPS Results

Mean unemployment **duration** rises monotonically with age (even though the unemployment *rate* falls with age):

| Age group | Mean unemployment duration |
|---|---:|
| `<30` | 21.9 weeks |
| `30-39` | 27.2 weeks |
| `40-49` | 30.0 weeks |
| `50+` | 32.9 weeks |

![Smoothed unemployment duration by age group](reports/figures/cps/unemployment_duration_age_group_smoothed.png)

*Figure 1. Smoothed unemployment duration by age group. Older workers face longer re-employment durations across most years.*

This does not show that screening models cause older-worker disadvantage; it establishes that the disadvantage exists in the broader market. The experiments then test whether screening systems reproduce that pattern through proxy features and reference-population effects.

---

## The Synthetic Testbed

![Data generation pipeline](reports/diagrams/data_generation_pipeline.png)

*Figure 2. Synthetic data generation, label construction, and downstream evaluation pipeline.*

The structured dataset is **50,000 résumés** with known age, a 51-feature canonical schema, and callback labels generated under a transparent scoring function. Age is never an input to any screening model.

Two design features make the causal reading clean:

- **A built-in negative control.** The same résumés are re-labeled under four bias settings that share a **byte-identical feature matrix** (features are keyed on a generation seed; only the label's *use* of the features changes). The `negative_control` setting has no callback disparity by construction, so any disparity a model shows under it is a measurement artifact, not a real effect.
- **A 40,000 / 10,000 train/test split.** All by-age tables in Notebook 03 onward are computed on the 10,000-row held-out test set (≈ 2,494 / 3,001 / 2,501 / 1,494 / 510 across the five age bands).

---

## Experiment A — Supervised Screening: Transmit, Not Amplify

Two screening models were trained to predict callbacks from structured features, with age excluded.

| Metric | Logistic regression | Gradient boosting |
|---|---:|---:|
| Accuracy | 0.871 | 0.877 |
| ROC AUC | 0.951 | 0.954 |

Outcomes vary sharply by age. The model neither invents nor magnifies the gap — it **transmits** the disparity already present in the labels:

| Age group | Actual callback rate | Predicted selection rate (logreg) | True-positive rate | False-negative rate |
|---|---:|---:|---:|---:|
| `<30` | 61.6% | 63.8% | 89.8% | 10.2% |
| `30-39` | 65.5% | 65.2% | 89.6% | 10.4% |
| `40-49` | 47.3% | 44.4% | 82.8% | 17.2% |
| `50-59` | 19.3% | 20.0% | 75.3% | 24.7% |
| `60+` | 5.3% | 7.3% | 59.3% | 40.7% |

![Selection rate by age group](reports/figures/selection_rate_by_age_group.png)

![True-positive rate by age group](reports/figures/tpr_by_age_group.png)

*Figures 3–4. Selection rate and true-positive rate by age group at 50k.*

### Disparate impact and the negative control

Against the strongest band, the four-fifths (0.80) rule is breached for the three oldest groups: disparate-impact ratios of **0.682 (40-49)**, **0.307 (50-59)**, and **0.111 (60+)**; equal-opportunity difference **0.303**.

The negative control confirms these are not pipeline artifacts. Re-running the identical pipeline on the no-disparity labels collapses every flag:

| Quantity | Canonical labels | Negative control |
|---|---:|---:|
| Selection-rate parity gap (max − min) | 0.579 | 0.034 |
| Minimum disparate-impact ratio | 0.111 | 0.974 |
| Bands flagged under the four-fifths rule | 3 | 0 |
| Equal-opportunity difference | 0.303 | 0.040 |

**Reading.** A single decision should be reported from the model that produced it: the logistic model slightly **over-predicts** the `60+` selection rate (7.3% vs an actual 5.3%) — the disadvantage is inherited from the labels, not added by the model. (Per-band counts in the oldest group are small, so the `60+` line is read with its confidence interval.)

---

## Experiment B — The Graduation-Year Ablation and the Proxy Cluster

| Variant | Features | ROC AUC | `30-39` − `60+` selection gap |
|---|---:|---:|---:|
| With `graduation_year` | 51 | 0.951 | 0.579 |
| Without `graduation_year` | 50 | 0.951 | ≈ same (CIs overlap at every band) |
| Without the full proxy cluster | 40 | 0.867 | **−0.007** |

Removing `graduation_year` changes nothing: the with- and without-`graduation_year` selection rates have **overlapping Wilson intervals at every age band**, so the two proxies are mutually substitutable. Removing the entire correlated cluster (graduation year, experience, tenure, title, tech mix, salary) is what collapses the disparity — the selection rate flattens to ≈ 0.50 across all ages and every four-fifths flag clears — and it costs real accuracy (AUC 0.951 → 0.867).

An incremental ablation makes the redundancy explicit: dropping the proxies one at a time holds the gap near 0.58 → 0.56 through the first seven removals; it only collapses after the eleventh feature (salary) is removed. No single feature is load-bearing — each dropped proxy is absorbed by the others.

![Ablation: predicted selection rate by age](reports/figures/ablation_predicted_rate_by_age.png)

![Ablation: incremental gap and AUC](reports/figures/ablation_incremental_gap_auc.png)

*Figures 5–6. The disparity survives single-proxy removal and collapses only when the correlated cluster is removed wholesale, at a clear accuracy cost.*

---

## Experiment C — Age Recoverability (and Why It Is Not the Same as Disparity)

Models trained to predict **age group** from résumé features (age excluded) recover it far above the 30% majority-class baseline:

| Feature set | Logreg accuracy | Gradient-boosting accuracy | Lift over baseline |
|---|---:|---:|---:|
| With `graduation_year` | 0.941 | 0.938 | ≈ 3.1× |
| Without `graduation_year` | 0.900 | 0.902 | ≈ 3.0× |
| Without `graduation_year` **and** experience | 0.850 | 0.846 | ≈ 2.8× |

Age stays ≈ 3× recoverable without `graduation_year`, and dropping the two experience columns on top of it still leaves it ≈ 2.8× recoverable — the same redundant encoding seen in Experiment B, from the recoverability side.

But recoverability is **necessary, not sufficient**, for a disparity — and this can be shown as an exact identity. Because the bias settings share a byte-identical feature matrix, age is recoverable in the `negative_control` data to the **same accuracy, to the digit**:

```
age recoverability (logreg, without graduation_year)
  canonical        : 0.9001
  negative control : 0.9001
  difference       : 0.00e+00
```

Age is exactly as predictable where there is **no** callback disparity. Predictability sets up the leak; the scoring function decides whether it flows.

---

## Experiment D — Structured Similarity Screening: The Reference Swap

![Similarity-based screening](reports/diagrams/similarity_experiment.png)

*Figure 7. Similarity screening disadvantages candidates who do not resemble the reference population — but the direction is a property of the reference, not the résumés.*

Applicants were scored by cosine similarity to a reference pool, with no training labels. Against a conventional "successful employee" reference (**R1** — mid-experience, senior titles, modern ≥ legacy tech), which is **100% under 50 by construction**, similarity falls steeply with age and the oldest bands are screened out entirely:

| Age group | Similarity vs R1 (mid-career) | Screen-pass rate vs R1 | Similarity vs R2 (age-balanced) |
|---|---:|---:|---:|
| `<30` | 0.551 | 94.4% | 0.090 |
| `30-39` | 0.452 | 54.0% | 0.246 |
| `40-49` | 0.205 | 1.2% | 0.441 |
| `50-59` | −0.029 | 0.0% | 0.550 |
| `60+` | −0.167 | 0.0% | 0.580 |

Scoring the **same applicants** against an age-balanced reference (**R2**) **reverses the gradient** — older candidates now score highest. The disparity direction is set by who is in the reference pool, not by the résumés.

![Reference-composition swap](reports/figures/similarity_reference_swap.png)

*Figure 8. The R1 → R2 reference swap reverses the age gradient on identical applicants.*

(A matched-pair check — building a "senior" variant by pushing every feature away from the mid-career template — finds it less similar in 100% of pairs, mean delta −0.332. This is a geometric near-tautology and is reported only as a monotonicity sanity check, not as evidence. Removing `graduation_year` shifts similarity by a mean of 0.019.)

---

## Experiment E — Full-Text Screening Against a Job-Description Ladder

Structured résumés were rendered into full text under factual-preservation constraints (an independent text-level audit checks for placeholder artifacts, leaked field codes, and explicit age language; a faithfulness/text-clean gate drops 74 of the rendered résumés, and the result is unchanged with or without it). The full-text corpus holds 12,500 résumés per generation mode.

Scoring the **same** résumés against a ladder of job descriptions, the favored age band climbs with the JD's seniority:

| Job description | Favored (highest-similarity) age band |
|---|---|
| Junior SWE | `30-39` |
| Mid SWE | `30-39` |
| Senior SWE | `40-49` |
| Staff SWE | `40-49` |
| Engineering Manager | `60+` |

Against the manager JD, the screen-in rate rises with age — **24.6% / 40.8% / 52.8% / 52.0% / 54.9%** (Wilson CIs, ≈ 2,490 per band; population-reweighted overall rate 0.421). The gradient is correlated with résumé length (r = 0.20) but survives length adjustment (length-adjusted similarity still climbs monotonically, −0.015 → +0.008).

![JD seniority sweep](reports/figures/nlp07_jd_seniority_sweep.png)

*Figure 9. The favored age band marches up the age range as the job description gets more senior.*

The within-notebook two-target contrast is the key point: the **same résumés** that the manager-JD screen favors with age (24.6% → 54.9%) are penalized with age by the structured callback model (59.5% → 5.4%). Opposite gradients, same people — the screening target decides the direction.

---

## Experiment F — Full-Text Incumbent Similarity, Read Through the Anisotropy Correction

Notebook 08 scores the same full-text résumés against an **incumbent reference pool**, swapping a mid-career reference (R1, again 100% under 50) for an age-balanced one (R2). The honest headline is a methodological one.

**On raw cosine, the swap washes out.** Same-domain résumé embeddings are anisotropic — pairwise cosines cluster near 0.8 — so the R1 and R2 centroids are nearly collinear and both targets peak at `40-49` (similarity 0.78–0.81 across all ages; corr(similarity, tokens) = 0.53, i.e. largely length-driven). A naïve raw-cosine incumbent screen would look roughly **age-neutral**.

**Under the standard mean-centering correction, the swap reappears and reverses,** replicating Experiment D:

| Age group | Centered similarity vs R1 | Centered swap effect (R2 − R1) |
|---|---:|---:|
| `<30` | 0.074 | −0.076 |
| `30-39` | 0.081 | −0.082 |
| `40-49` | −0.018 | +0.016 |
| `50-59` | −0.056 | +0.062 |
| `60+` | −0.069 | +0.076 |

R1 favors the young/mid bands; switching to the balanced reference helps older candidates and hurts younger ones. The masking is itself the finding: a real incumbent screen can look fair on raw cosine while the age-composition bias sits latent in the representation, surfacing under ordinary normalization.

![Incumbent reference swap](reports/figures/nlp08_reference_swap.png)

*Figure 10. Raw cosine masks the incumbent swap; mean-centering recovers the R1 → R2 reversal.*

---

## Experiment G — External Validation on Real Résumé Text

Notebook 09 applies the framework to a real Kaggle résumé corpus (9,000 rows → **5,514 technical résumés** after filtering and de-duplication). It has **no true age labels**, so it uses **inferred career-stage proxies** (stated years, title seniority): early 80 / mid 2,718 / senior 1,303 / very-senior 213 / unknown 1,200, with 65% stating years of experience. This is an external-validity check, explicitly secondary to the controlled experiments; a weak result is informative, not a failure.

**The job-description mechanism replicates cleanly.** The favored career stage climbs as the JD goes junior → manager (junior/mid JD → `mid_career`; senior/staff/manager JD → `very_senior`), and the manager-JD screen-in rate rises monotonically with **non-overlapping** Wilson intervals:

| Career stage | Manager-JD screen rate | 95% CI |
|---|---:|---|
| `early_career` | 31.2% | [22.2%, 42.1%] |
| `mid_career` | 43.7% | [41.9%, 45.6%] |
| `senior_career` | 56.9% | [54.2%, 59.5%] |
| `very_senior` | 70.4% | [64.0%, 76.1%] |

This is **not** a length artifact (corr with tokens = 0.02; length-adjusted similarity still climbs) and **not** a legacy-stack artifact (the manager-JD similarity barely moves across legacy-tech tertiles, r = 0.07, while modern-tech presence tracks it far more, r ≈ 0.26).

**The incumbent mechanism does not cleanly transfer — reported as inconclusive.** On raw cosine it washes out (anisotropy again). The centered swap effect is positive across stages (+0.01 → +0.06), seeming to replicate — but it is confounded by the corpus's heavy mid-career concentration, and once the applicant pool is stage-balanced the sign **flips** for the senior stages (balanced swap effect +0.06 / +0.18 / −0.03 / −0.07). The reference-swap result therefore holds firmly only in the controlled synthetic setting; on this real corpus it is not established.

---

## Consolidated Controls

Each mechanism is paired with a control that could have falsified it:

| Control | What it tests | Result |
|---|---|---|
| **Negative control** (03) | Is the disparity real or a pipeline artifact? | Parity gap 0.579 → 0.034; min DI 0.111 → 0.974; 3 flags → 0; EOD 0.303 → 0.040. Real. |
| **Recoverability identity** (05) | Does predictability of age imply a disparity? | Age recoverable to the same accuracy (0.9001) with or without a disparity. No — necessary, not sufficient. |
| **Cluster ablation** (04) | Is one proxy responsible? | Single-proxy removal: no change (CIs overlap). Whole-cluster removal: gap → −0.007, at AUC 0.951 → 0.867. Redundant, distributed. |
| **Reference swap, structured** (06) | Is the direction a property of the résumés? | R1 → R2 reverses the gradient on identical applicants. No — it is a property of the reference. |
| **JD seniority sweep** (07) | Does the screening target set the direction? | Favored band climbs junior → manager; same résumés show the opposite gradient under the callback model. Yes. |
| **Incumbent swap + anisotropy** (08) | Does the swap survive in text embeddings? | Washed out on raw cosine; recovered and reversed under mean-centering. Yes, once anisotropy is corrected. |
| **External validation** (09) | Do the mechanisms appear on real text? | JD mechanism replicates (non-overlapping CIs, length- and stack-independent); incumbent mechanism inconclusive (sign flips under stage-balancing). |

---

## Consolidated Findings

1. **Explicit age is not required, and the model does not amplify the gap — it transmits it.** Predicted selection rates track the (biased) label rates; the oldest band is, if anything, slightly over-predicted.
2. **Removing one proxy does nothing.** `graduation_year` and experience are mutually substitutable with the rest of a correlated cluster; only removing the whole cluster collapses the disparity, and at a real accuracy cost.
3. **Age is ≈ 3× recoverable from ordinary résumé content** even with age and its obvious proxies removed — but recoverability is necessary, not sufficient: it is identical where there is no disparity at all.
4. **The screening target sets the direction.** A mid-career incumbent reference penalizes older candidates; an age-balanced reference reverses it; a senior job description favors them. The same résumés flip sign across targets.
5. **Embedding anisotropy can mask the effect.** On raw cosine an incumbent screen can look age-neutral while the bias sits latent in the representation, surfacing under standard normalization.
6. **The job-description mechanism replicates on real résumé text; the incumbent mechanism does not cleanly transfer** and is reported as inconclusive on the external corpus.

---

## Hypothesis (Updated)

> Older workers already face longer re-employment durations. Résumé-screening systems reproduce and **redistribute** that disadvantage — rather than amplifying it — whenever job-relevant features proxy for age and a reference population or screening target encodes an age composition. The direction of the resulting disparity is determined by the target, not by the résumés.

---

## Conclusion

The project does not claim that every résumé-screening model disadvantages older candidates. The defensible claim is narrower and more useful: such systems can produce age disparities through several distinct channels — supervised transmission of biased labels, redundant proxy leakage across correlated features, reference-population similarity, incumbent-profile targets, and embedding behavior that depends on the comparison target — and the direction of the disparity is a property of the target, not the applicants.

The practical implication is that dropping protected attributes or one obvious proxy is not a sufficient control. Screening systems need to be audited across age groups, the full proxy cluster, reference-set construction, target definition, screening thresholds, and the embedding normalization used to compute similarity.

---

## Notes and Limitations

- Synthetic data is used for controlled mechanism testing; the CPS provides labor-market context, not causal evidence about screening systems.
- The external Kaggle corpus has no true age labels; the career-stage proxies are not protected attributes, and that notebook is small-N.
- Similarity results depend on the embedding model, reference-set composition, thresholding, and — as Experiment F shows — the anisotropy correction applied.
- The incumbent-swap mechanism is established only in the controlled synthetic setting.
- The study addresses age-related proxy bias and does not evaluate intersectional effects.

---

## Reproducing the Analysis

### Structured experiments (Notebooks 01–06)

Run `01_data_generation.ipynb` first (produces the 50,000-row dataset and the negative-control variants under `data/experiments/`), then `02` through `06` in order. All structured notebooks import the canonical feature list from `src/features.py`.

### CPS data (Notebook 00)

The CPS extract is not included due to size. Create an account at IPUMS CPS, extract Basic Monthly CPS with variables `AGE`, `EMPSTAT`, `DURUNEMP`, `OCC2010`, `WTFINL`, `YEAR`, `MONTH`, save under `data/raw/ipums_cps/`, then run `00_cps_data_prep.ipynb`.

### Full-text and external experiments (Notebooks 07–09)

Notebooks 07–08 use cached embeddings of the generated full-text corpus. Notebook 09 uses the Kaggle Resume Dataset; save the CSV as `data/external/resume_dataset.csv`, then run `09_external_resume_career_stage_validation.ipynb`.

---

## Repository Outputs

Key outputs are written under `data/experiments/`, `reports/tables/`, and `reports/figures/`. Figures referenced above:

```text
reports/figures/cps/unemployment_duration_age_group_smoothed.png
reports/figures/selection_rate_by_age_group.png
reports/figures/tpr_by_age_group.png
reports/figures/ablation_predicted_rate_by_age.png
reports/figures/ablation_incremental_gap_auc.png
reports/figures/similarity_reference_swap.png
reports/figures/nlp07_jd_seniority_sweep.png
reports/figures/nlp08_reference_swap.png
reports/diagrams/data_generation_pipeline.png
reports/diagrams/similarity_experiment.png
```

Notebook 09's results are reported as tables (`reports/tables/ext09_*.csv`) rather than figures.
