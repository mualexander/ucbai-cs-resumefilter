# Resume Screening Proxy Bias Experiment

## Overview

This project investigates whether machine learning models used in resume screening can learn and reproduce age-related disparities **without being explicitly provided age information**, through the use of correlated proxy variables such as:

- Years of experience
- Graduation year
- Title seniority
- Salary expectations
- Legacy vs. modern technology signals

The core objective is to demonstrate how proxy feature leakage can produce measurable disparities in model outcomes even when protected attributes are excluded from training.

---

## Research Question

Can a resume-screening model produce systematically different outcomes for older candidates due to proxy features, even when age is not included in the feature set?

---

## Experimental Design

1. **Synthetic Resume Generation**
   - Realistic correlations between latent age group and resume features.
   - Adjustable bias strength in simulated hiring outcome.
   - Fully reproducible via fixed seed and metadata.

2. **Model Training**
   - Logistic Regression (baseline)
   - Optional tree-based models
   - Age and age_group excluded from feature set

3. **Fairness Evaluation**
   - Callback rate by age group
   - False Positive / False Negative rates
   - Statistical Parity Difference
   - Disparate Impact Ratio
   - Equal Opportunity Difference

4. **Ablation Study**
   - Compare models trained:
     - With `graduation_year`
     - Without `graduation_year`
   - Measure persistence of disparity

---

## Repository Structure
resume-screening-proxy-bias/
│
├── data/
│ ├── baseline/
│ └── experiments/
│
├── notebooks/
│ ├── resume_screening_proxy_bias_experiment.ipynb
│ ├── 01_data_generation.ipynb
│ ├── 02_model_training.ipynb
│ ├── 03_fairness_evaluation.ipynb
│ └── 04_ablation_graduation_year.ipynb
│
├── models/
├── reports/
│ ├── figures/
│ └── tables/
│
└── src/


---

## Reproducibility

The synthetic dataset is generated using:

- Fixed random seed
- Explicit bias strength parameter
- Stored generation metadata (`generation_metadata.json`)

This allows exact replication of experiments.

---

## Key Insight Being Tested

Even when age is removed from training data, machine learning systems may still:

- Infer age from correlated features
- Encode that signal in model weights
- Produce unequal outcomes across age groups

This experiment isolates and measures that effect.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt

Primary libraries:

pandas

numpy

scikit-learn

matplotlib

seaborn

pyarrow

Status

Initial experiment framework implemented:

Synthetic data generator

Baseline model training

Graduation year ablation scaffolded

Fairness metric implementation and bias sweep experiments in progress.

