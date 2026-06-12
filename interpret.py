import numpy as np
from sklearn.ensemble import RandomForestClassifier

data = np.load("features.npz")
X, y, groups = data["X"], data["y"], data["groups"]

mask = np.isin(y, ["A", "C"])
X, y = X[mask], y[mask]
y = (y == "A").astype(int)          # 1 = AD, 0 = control

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
n_ch = 19                            # columns are ordered band-by-band: band b, channel c -> b*19 + c

# 1. Clinical check: is AD's spectrum actually slowed?  (delta/theta up, alpha down)
print("Mean relative band power (averaged over channels & epochs):")
print(f"{'band':<7}{'control':>10}{'AD':>8}{'AD - ctrl':>12}")
for b, name in enumerate(BANDS):
    lo, hi = b * n_ch, (b + 1) * n_ch
    ctrl = X[y == 0][:, lo:hi].mean()
    ad   = X[y == 1][:, lo:hi].mean()
    print(f"{name:<7}{ctrl:>10.3f}{ad:>8.3f}{ad - ctrl:>+12.3f}")

# 2. Model check: which bands does the forest actually rely on?
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X, y)                         # fit on all binary data — only to read importances, not to score
band_imp = rf.feature_importances_.reshape(5, n_ch).sum(axis=1)
band_imp /= band_imp.sum()

print("\nForest importance by band (fraction of total):")
for b in np.argsort(band_imp)[::-1]:
    print(f"{BANDS[b]:<7}{band_imp[b]:>8.1%}")