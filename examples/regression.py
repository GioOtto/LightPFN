"""Regression mean and quantile intervals, without a training step."""
from sklearn.datasets import load_diabetes
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from lightpfn import LightPFNRegressor

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
reg = LightPFNRegressor(device="cpu", random_state=0).fit(Xtr, ytr)
print("RMSE:", root_mean_squared_error(yte, reg.predict(Xte)))
print("10/50/90% quantiles:", reg.predict_quantiles(Xte[:5], [.1, .5, .9]))
