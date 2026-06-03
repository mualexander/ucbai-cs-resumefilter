"""Canonical baseline model definitions.

Single source of truth for notebook 02 (training) and notebook 03's
negative control, so both fit *identical* model classes and only the
data/label differs — which is what makes the negative control a clean control.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from src.preprocessing import make_preprocessor


def make_baseline_models(X, random_state=42):
    """Return the canonical {name: unfitted Pipeline} baseline models.

    Pass the feature frame as X; only its dtypes/columns are inspected to build
    the preprocessor, so the full frame and the train split give the same
    column partition. A fresh preprocessor is built per model so each pipeline
    owns its own fitted state — never share one ColumnTransformer across two
    pipelines, or fitting the second silently re-fits the first.
    """
    return {
        "logreg": Pipeline([
            ("preprocessor", make_preprocessor(X)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]),
        "hgb": Pipeline([
            ("preprocessor", make_preprocessor(X)),
            ("classifier", HistGradientBoostingClassifier(random_state=random_state)),
        ]),
    }