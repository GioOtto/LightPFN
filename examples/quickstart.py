"""Fit LightPFN on a small table and compare it with a default random forest.

    python examples/quickstart.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightpfn import LightPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# The first fit downloads the weights (18 MB) from Hugging Face and caches them.
clf = LightPFNClassifier(n_estimators=4, random_state=0).fit(X_train, y_train)
rf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

print(f"device: {clf.device_}")
print(f"LightPFN      AUC {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]):.4f}")
print(f"Random forest AUC {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")
