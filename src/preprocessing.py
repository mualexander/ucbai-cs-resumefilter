from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def split_feature_types(X):
    """Partition columns into (numeric, categorical, boolean); assert full coverage."""
    boolean = X.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    numeric = [c for c in X.select_dtypes(include=["number"]).columns if c not in boolean]
    categorical = X.select_dtypes(include=["category", "object", "string"]).columns.tolist()

    partitioned = set(numeric) | set(categorical) | set(boolean)
    expected = set(X.columns)
    missing = sorted(expected - partitioned)
    extra = sorted(partitioned - expected)
    assert not missing, f"Features not assigned to a preprocessing bucket: {missing}"
    assert not extra, f"Unexpected preprocessing features: {extra}"
    return numeric, categorical, boolean


def make_preprocessor(X):
    """Median/most-frequent imputation + scaling/one-hot. Returns a fresh, unfitted transformer."""
    numeric, categorical, boolean = split_feature_types(X)
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("bool", SimpleImputer(strategy="most_frequent"), boolean),
    ])