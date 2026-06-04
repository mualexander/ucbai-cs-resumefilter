import numpy as np


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion k/n. Returns (lo, hi)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mean_ci(x, z=1.96):
    """Mean with a normal-approximation 95% CI. Returns (mean, lo, hi).

    Shared by notebooks 07/08 for mean-similarity intervals (previously two
    identical in-notebook copies).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean()) if n else np.nan
    h = z * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - h, m + h
