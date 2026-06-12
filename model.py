import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold, cross_val_score, cross_validate

data = np.load("features.npz")
X, y, groups = data["X"], data["y"], data["groups"]

# Binary task: Alzheimer's (A) vs healthy control (C). Park FTD (F) for now.
mask = np.isin(y, ["A", "C"])
X, y, groups = X[mask], y[mask], groups[mask]
y = (y == "A").astype(int)          # 1 = Alzheimer's, 0 = control

print(f"Binary AD-vs-control: {X.shape[0]} epochs from {len(np.unique(groups))} subjects")
print(f"  AD epochs: {(y==1).sum()}  |  Control epochs: {(y==0).sum()}")

# Subject-wise cross-validation — the rigor guard.
cv = GroupKFold(n_splits=5)

# Baseline: always guess the majority class. This is what 'chance' really is.
dummy = DummyClassifier(strategy="most_frequent")
base = cross_val_score(dummy, X, y, groups=groups, cv=cv, scoring="accuracy")

# The model.
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
res = cross_validate(rf, X, y, groups=groups, cv=cv,
                     scoring=["accuracy", "balanced_accuracy"])
acc, bal = res["test_accuracy"], res["test_balanced_accuracy"]

print()
print(f"Baseline (majority class) accuracy : {base.mean():.3f}")
print(f"Random forest accuracy             : {acc.mean():.3f} ± {acc.std():.3f}")
print(f"Random forest balanced accuracy    : {bal.mean():.3f} ± {bal.std():.3f}")