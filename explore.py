from pathlib import Path
import mne
import pandas as pd
import matplotlib.pyplot as plt

BIDS_ROOT = Path("data/ds004504")   # adjust if you downloaded elsewhere

# 1. Labels live in participants.tsv, not the EEG files.
participants = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t")
print(participants.head(), "\n")
print("Columns:", list(participants.columns), "\n")
for col in participants.columns:
    if participants[col].nunique() < 6:
        print(f"  {col} -> {participants[col].unique()}")

# 2. Load one subject's raw EEG (let MNE find the .set file).
sub_dir = sorted(BIDS_ROOT.glob("sub-*"))[0]
set_file = next(sub_dir.glob("eeg/*.set"))
print(f"\nLoading {set_file.name}")
raw = mne.io.read_raw_eeglab(set_file, preload=True)

print(raw.info)
print("Sampling rate:", raw.info["sfreq"], "Hz")
print("Duration:", round(raw.times[-1], 1), "s")
print("Channels:", raw.ch_names)

# 3. The spectrum — the alpha peak (~8-12 Hz) is what AD slowing eats into.
raw.compute_psd(fmax=45).plot()
plt.show()