from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")               # write files, never open a blocking window
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

OUT = Path("figures"); OUT.mkdir(exist_ok=True)

data = np.load("features.npz")
X, y = data["X"], data["y"]
mask = np.isin(y, ["A", "C"])
X, y = X[mask], y[mask]
y = (y == "A").astype(int)

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
n_ch = 19

# Figure 1: spectral slowing, AD vs control
ctrl = [X[y == 0][:, b*n_ch:(b+1)*n_ch].mean() for b in range(5)]
ad   = [X[y == 1][:, b*n_ch:(b+1)*n_ch].mean() for b in range(5)]
xpos, w = np.arange(5), 0.38
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(xpos - w/2, ctrl, w, label="control", color="#1D9E75")
ax.bar(xpos + w/2, ad,   w, label="Alzheimer's", color="#D85A30")
ax.set_xticks(xpos); ax.set_xticklabels(BANDS)
ax.set_ylabel("mean relative band power")
ax.set_title("Spectral slowing in AD: more delta/theta, less alpha")
ax.legend(); fig.tight_layout()
fig.savefig(OUT / "band_power_AD_vs_control.png", dpi=150)

# Figure 2: which bands the model relies on
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X, y)
imp = rf.feature_importances_.reshape(5, n_ch).sum(axis=1)
imp = imp / imp.sum()
order = np.argsort(imp)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([BANDS[i] for i in order], imp[order], color="#534AB7")
ax.set_ylabel("fraction of total importance")
ax.set_title("Random forest band importance")
fig.tight_layout()
fig.savefig(OUT / "band_importance.png", dpi=150)

print("Saved figures/band_power_AD_vs_control.png and figures/band_importance.png")