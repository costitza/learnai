import numpy as np
import pandas as pd

# 1. Load both submission CSVs (update paths to match your exact filenames)
sub_robert = pd.read_csv("submission_robert.csv")       # Fine-tuned Romanian RoBERT
sub_rogpt2 = pd.read_csv("submission_rogpt2_calibrated.csv") # RoGPT2-base surprisal + tree/Ridge

# 2. Align strictly by datapointID (prevents any row-order mismatch bugs)
merged = pd.merge(
    sub_robert[["subtaskID", "datapointID", "answer"]],
    sub_rogpt2[["datapointID", "answer"]],
    on="datapointID",
    suffixes=("_robert", "_rogpt2")
)

# 3. Weighted Blend (give slightly more weight to your higher-scoring model)
w_robert = 0.45
w_rogpt2 = 0.55

merged["answer"] = np.clip(
    w_robert * merged["answer_robert"] + w_rogpt2 * merged["answer_rogpt2"],
    0,
    None
)

# 4. Format & Save Final Ensemble Submission
sub_ensemble = merged[["subtaskID", "datapointID", "answer"]].copy()
sub_ensemble["answer"] = np.round(sub_ensemble["answer"], 4)

sub_ensemble.to_csv("submission_ensemble.csv", index=False)

# 5. Mandatory Sanity Checks Before Uploading to Nitro Judge
print("Shape:", sub_ensemble.shape, "| Expected rows:", len(sub_robert))
print("Columns:", sub_ensemble.columns.tolist())
print("Null count:", sub_ensemble.isna().sum().sum())
print("Correlation between the 2 models:", merged["answer_robert"].corr(merged["answer_rogpt2"]))
print(sub_ensemble.head(10))