"""Build the executive-summary hero figure from saved tables.

Left panel : predicted callback rate by age, three ablation arms (notebook 04)
Right panel: full-text screen-in rate by age under two targets (notebooks 07/08)

Run from anywhere inside the repo:  python scripts/make_hero_figure.py
Reads only reports/tables/*.csv; writes reports/figures/hero_proxy_cluster_and_target.png
"""
import sys
from pathlib import Path

_root = Path.cwd().parent if Path.cwd().name in ("notebooks", "scripts") else Path.cwd()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.paths import TABLES_DIR, FIGURES_DIR, rel
from src.features import AGE_ORDER

ARM_LABELS = {
    "with_grad_year": "all features",
    "without_grad_year": "without graduation_year",
    "without_proxy_cluster": "without full proxy cluster",
}

# ---- left panel data: three-arm selection rates with Wilson CIs (notebook 04) ----
ci = pd.read_csv(TABLES_DIR / "ablation_three_arm_selection_ci.csv")

# ---- right panel data: full-text screen rates (notebooks 07 / 08) ----
jd  = pd.read_csv(TABLES_DIR / "nlp07_screen_rates_eng_manager.csv").set_index("age_group").reindex(AGE_ORDER)
inc = pd.read_csv(TABLES_DIR / "nlp08_screen_rates_R1_centered.csv").set_index("age_group").reindex(AGE_ORDER)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
x = np.arange(len(AGE_ORDER))

# Panel A — grouped bars, Wilson error bars
width = 0.27
for i, arm in enumerate(ARM_LABELS):
    g = ci[ci.arm == arm].set_index("age_group").reindex(AGE_ORDER)
    vals = g["selection_rate"].to_numpy()
    err = np.vstack([vals - g["wilson_lo"].to_numpy(), g["wilson_hi"].to_numpy() - vals])
    ax1.bar(x + (i - 1) * width, vals, width, yerr=err, capsize=2, label=ARM_LABELS[arm])
ax1.set_xticks(x); ax1.set_xticklabels(AGE_ORDER)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Age group"); ax1.set_ylabel("Predicted callback rate")
ax1.set_title("Structured screen: the proxy cluster carries the gap")
ax1.legend(fontsize=9)

# Panel B — two opposite gradients, Wilson error bars
for df, label, marker in ((jd, "engineering-manager job description (07)", "o"),
                          (inc, "mid-career incumbent profile, centered (08)", "s")):
    vals = df["screen_rate"].to_numpy()
    err = np.vstack([vals - df["lo"].to_numpy(), df["hi"].to_numpy() - vals])
    ax2.errorbar(x, vals, yerr=err, marker=marker, capsize=3, label=label)
ax2.set_xticks(x); ax2.set_xticklabels(AGE_ORDER)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Age group"); ax2.set_ylabel("Top-45% screen-in rate")
ax2.set_title("Full-text screens: the target sets the direction")
ax2.legend(fontsize=9)

fig.tight_layout()
out = FIGURES_DIR / "hero_proxy_cluster_and_target.png"
fig.savefig(out, dpi=150)
print("wrote", rel(out))
