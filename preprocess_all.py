from pathlib import Path
import mne
import pandas as pd
from mne.preprocessing import ICA
from mne_icalabel import label_components

mne.set_log_level("WARNING")          # quiet the per-step console spam

BIDS_ROOT = Path("data/ds004504")
OUT_DIR = Path("data/clean")          # gitignored (lives under data/)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SUBJECTS = None                       # 3 = test run; set to None for all 88

# subject id -> group label (A / C / F)
participants = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t").set_index("participant_id")

def preprocess_subject(set_file):
    raw = mne.io.read_raw_eeglab(set_file, preload=True)
    raw.filter(0.5, 45)
    raw.notch_filter(50)
    raw.set_eeg_reference("average")
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, overlap=0.0, preload=True)

    ica = ICA(n_components=15, method="infomax", fit_params=dict(extended=True),
              random_state=42, max_iter="auto")
    ica.fit(raw)
    ic = label_components(raw, ica, method="iclabel")
    ica.exclude = [i for i, (lab, p) in enumerate(zip(ic["labels"], ic["y_pred_proba"]))
                   if lab not in ("brain", "other") and p >= 0.80]
    return ica.apply(epochs)

subjects = sorted(BIDS_ROOT.glob("sub-*"))[:MAX_SUBJECTS]
done, failed = 0, []
for n, sub_dir in enumerate(subjects, 1):
    sub = sub_dir.name
    out_file = OUT_DIR / f"{sub}_clean-epo.fif"
    if out_file.exists():                          # resumable: skip finished subjects
        print(f"[{n}/{len(subjects)}] {sub} already done, skipping")
        done += 1
        continue
    try:
        group = participants.loc[sub, "Group"]
        print(f"[{n}/{len(subjects)}] {sub} ({group}) ...")
        epochs_clean = preprocess_subject(next(sub_dir.glob("eeg/*.set")))
        epochs_clean.save(out_file, overwrite=True)
        done += 1
    except Exception as e:
        print(f"  !! {sub} FAILED: {e}")
        failed.append(sub)

print(f"\nFinished: {done} done, {len(failed)} failed.")
if failed:
    print("Failed subjects:", failed)