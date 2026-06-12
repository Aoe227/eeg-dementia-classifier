from pathlib import Path
import mne
import matplotlib.pyplot as plt

BIDS_ROOT = Path("data/ds004504")

sub_dir = sorted(BIDS_ROOT.glob("sub-*"))[0]
set_file = next(sub_dir.glob("eeg/*.set"))
raw = mne.io.read_raw_eeglab(set_file, preload=True)

raw.filter(l_freq=0.5, h_freq=45)
raw.notch_filter(freqs=50)
raw.set_eeg_reference("average")
epochs = mne.make_fixed_length_epochs(raw, duration=4.0, overlap=0.0, preload=True)
print(epochs)
print("Shape (epochs, channels, samples):", epochs.get_data(copy=False).shape)

from mne.preprocessing import ICA
from mne_icalabel import label_components

ica = ICA(n_components=15, method="infomax", fit_params=dict(extended=True),
          random_state=42, max_iter="auto")
ica.fit(raw)

ic_labels = label_components(raw, ica, method="iclabel")
labels = ic_labels["labels"]
probs = ic_labels["y_pred_proba"]
for i, (lab, p) in enumerate(zip(labels, probs)):
    print(f"  {i:2d}  {lab:12s}  {p:.2f}")

exclude_idx = [i for i, (lab, p) in enumerate(zip(labels, probs))
               if lab not in ("brain", "other") and p >= 0.80]
print("Excluding:", exclude_idx)

ica.exclude = exclude_idx
epochs_clean = ica.apply(epochs.copy())
print(epochs_clean)