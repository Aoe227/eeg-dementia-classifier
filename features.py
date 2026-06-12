from pathlib import Path
import numpy as np
import pandas as pd
import mne

mne.set_log_level("WARNING")

CLEAN_DIR = Path("data/clean")
BIDS_ROOT = Path("data/ds004504")
participants = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t").set_index("participant_id")

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
         "beta": (12, 30), "gamma": (30, 45)}

def bandpower(epochs):
    spec = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)
    psds, freqs = spec.get_data(return_freqs=True)       
    total = psds.sum(axis=-1, keepdims=True)
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        rel = psds[:, :, mask].sum(axis=-1) / total[:, :, 0]   # relative band power
        feats.append(rel)
    return np.concatenate(feats, axis=1)                  

X_list, y_list, groups_list = [], [], []
for f in sorted(CLEAN_DIR.glob("sub-*_clean-epo.fif")):
    sub = f.name.split("_")[0]
    group = participants.loc[sub, "Group"]
    epochs = mne.read_epochs(f, preload=True, verbose=False)
    feats = bandpower(epochs)
    X_list.append(feats)
    y_list += [group] * len(feats)        
    groups_list += [sub] * len(feats)        
    print(f"{sub} ({group}): {feats.shape[0]} epochs")

X = np.vstack(X_list)
y = np.array(y_list)
groups = np.array(groups_list)
print("\nFeature matrix:", X.shape, "| labels:", y.shape, "| groups:", groups.shape)

np.savez("features.npz", X=X, y=y, groups=groups)
print("Saved features.npz")