from pathlib import Path

# Resolve from this file's location, not the notebook's cwd — robust to where you run from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # src/paths.py -> repo root

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_DIR = DATA_DIR / "baseline"
REPORTS_DIR = ROOT / "reports"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
EMBEDDINGS_DIR = EXPERIMENTS_DIR / "embeddings"
INTERIM_DIR = DATA_DIR / "interim"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

for d in (
    DATA_DIR,
    MODELS_DIR,
    BASELINE_DIR,
    REPORTS_DIR,
    EXPERIMENTS_DIR,
    EMBEDDINGS_DIR,
    INTERIM_DIR,
    TABLES_DIR,
    FIGURES_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

def rel(p):
    """Path relative to PROJECT_ROOT for clean, machine-independent printing."""
    p = Path(p)
    try:
        return p.relative_to(PROJECT_ROOT)
    except ValueError:
        return p   # outside the repo: print as-is rather than crashing a save cell