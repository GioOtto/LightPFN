"""LightPFN inside scikit-learn tools: cross-validation and a grid search over the number of estimators.

    python examples/sklearn_pipeline.py
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from lightpfn import LightPFNClassifier

X, y = load_digits(return_X_y=True)  # 1,797 images of 8 x 8 pixels, 10 classes (the most LightPFN supports)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

scores = cross_val_score(LightPFNClassifier(random_state=0), X, y, cv=cv, scoring="accuracy")
print(f"5-fold accuracy: {scores.mean():.4f} +- {scores.std():.4f}")

search = GridSearchCV(LightPFNClassifier(random_state=0), {"n_estimators": [1, 4]}, cv=cv, scoring="neg_log_loss")
search.fit(X, y)
print("best n_estimators:", search.best_params_["n_estimators"], f"log loss {-search.best_score_:.4f}")
