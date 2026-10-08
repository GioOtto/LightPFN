"""A pandas DataFrame with string, categorical, boolean and missing values, passed as it is.

String, category, object and bool columns become ordinal codes of the categories seen in fit; missing values
and categories not seen in fit become NaN, which the model reads as missing.

    python examples/pandas_categorical.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightpfn import LightPFNClassifier

rng = np.random.default_rng(0)
n = 1500
df = pd.DataFrame({
    "income": rng.lognormal(10, 0.6, n),
    "age": rng.integers(18, 80, n).astype(float),
    "plan": rng.choice(["basic", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1]),
    "region": pd.Categorical(rng.choice(["north", "south", "east", "west"], n)),
    "newsletter": rng.random(n) < 0.4,
})
df.loc[rng.random(n) < 0.1, "age"] = np.nan  # missing numbers
df.loc[rng.random(n) < 0.05, "plan"] = None  # missing categories
logit = (2.0 * (df["plan"] == "enterprise") + 1.5 * (df["region"] == "south") - 1.0 * df["newsletter"]
         - 0.06 * (df["age"].fillna(50) - 50) + np.log(df["income"] / 22000))
y = np.where(rng.random(n) < 1 / (1 + np.exp(-(logit - 0.5))), "churn", "stay")

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, stratify=y, random_state=0)
clf = LightPFNClassifier(n_estimators=4, random_state=0).fit(X_train, y_train)

print("categorical columns:", [c for c, flag in zip(df.columns, clf.is_categorical_) if flag])
print("categories of 'plan':", list(clf.categories_[2]))
print("classes:", clf.classes_)
print(f"LightPFN AUC {roc_auc_score(y_test == 'stay', clf.predict_proba(X_test)[:, 1]):.4f}")

# reference: scikit-learn's gradient boosting with native categorical support
codes = {c: X_train[c].astype("category").cat.categories for c in ["plan", "region"]}
encode = lambda f: f.assign(**{c: pd.Categorical(f[c], categories=codes[c]).codes for c in codes})
hgb = HistGradientBoostingClassifier(categorical_features=[2, 3], random_state=0).fit(encode(X_train), y_train)
print(f"HistGradientBoosting AUC {roc_auc_score(y_test == 'stay', hgb.predict_proba(encode(X_test))[:, 1]):.4f}")
