import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ================= 1. DATELE ȘI MAPĂRILE =================
train_df = pd.read_csv("../train_data.csv")
test_df = pd.read_csv("../test_data.csv")

EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}
EVASION_LABELS = list(EVASION_TO_CLARITY.keys())
CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

label2id = {lbl: i for i, lbl in enumerate(EVASION_LABELS)}
id2label = {i: lbl for i, lbl in enumerate(EVASION_LABELS)}
clarity2id = {lbl: i for i, lbl in enumerate(CLARITY_LABELS)}

y_evasion = train_df["evasion_label"].map(label2id).values
y_clarity = train_df["evasion_label"].map(EVASION_TO_CLARITY).map(clarity2id).values

def probs9_to_probs3(probs9):
    p_clear = probs9[:, [0]].sum(axis=1)
    p_ambiv = probs9[:, [1, 2, 3, 4, 5]].sum(axis=1)
    p_nonrep = probs9[:, [6, 7, 8]].sum(axis=1)
    return np.stack([p_clear, p_ambiv, p_nonrep], axis=1)

# ================= 2. ÎNCĂRCĂM FIȘIERELE .NPY =================
# --- Implementarea TA ---
my_oof_t1 = np.load("oof_sub1_rb3.npy")              # (N_train, 3)
my_test_t1 = np.load("test_sub1_rb3.npy")            # (N_test, 3)

# --- Implementarea COLEGULUI ---
col_oof_t1 = np.load("oof3_RoBERT-large.npy")        # (N_train, 3)
col_test_t1 = np.load("test3_RoBERT-large.npy")      # (N_test, 3)

col_oof_t2 = np.load("oof_RoBERT-large.npy")         # (N_train, 9)
col_test_t2 = np.load("test_RoBERT-large.npy")       # (N_test, 9)

# Normalizare de siguranță în caz că vreun fișier a fost salvat nescalat (/ 5.0)
for arr in [my_oof_t1, my_test_t1, col_oof_t1, col_test_t1, col_oof_t2, col_test_t2]:
    if arr[0].sum() > 1.5:
        arr /= 5.0

# ================= 3. ALINIEREA TESTULUI COLEGULUI DUPĂ DATAPOINTID =================
# Colegul a sortat test_df după datapointID înainte să salveze .npy-urile; le aducem la ordinea din test_data.csv
sort_dp = test_df.sort_values("datapointID").index.values
inv_dp = np.argsort(sort_dp)

col_test_t1_aligned = col_test_t1[inv_dp]
col_test_t2_aligned = col_test_t2[inv_dp]

# ================= 4. VERIFICARE SCORURI INDIVIDUALE & CORELAȚII =================
s1_mine = f1_score(y_clarity, np.argmax(my_oof_t1, axis=1), average="macro")
s1_col = f1_score(y_clarity, np.argmax(col_oof_t1, axis=1), average="macro")
s2_col = f1_score(y_evasion, np.argmax(col_oof_t2, axis=1), average="macro")

print(f"Task 1 - Modelul TĂU     : {s1_mine:.4f}")
print(f"Task 1 - Modelul COLEG   : {s1_col:.4f}")
print(f"Task 2 - Modelul COLEG   : {s2_col:.4f}")
print(f"Corelație OOF Task 1     : {np.corrcoef(my_oof_t1[:, 1], col_oof_t1[:, 1])[0, 1]:.4f}")
print(f"Corelație TEST Aliniată  : {np.corrcoef(my_test_t1[:, 1], col_test_t1_aligned[:, 1])[0, 1]:.4f}")

# ================= 5. BLENDING OPTIM PE TASK 1 =================
col_oof_t1_full = 0.70 * col_oof_t1 + 0.30 * probs9_to_probs3(col_oof_t2)
col_test_t1_full = 0.70 * col_test_t1_aligned + 0.30 * probs9_to_probs3(col_test_t2_aligned)

best_w, best_s1 = 0.5, 0.0
for w in np.linspace(0, 1, 21):
    blend_oof1 = w * my_oof_t1 + (1 - w) * col_oof_t1_full
    sc = f1_score(y_clarity, np.argmax(blend_oof1, axis=1), average="macro")
    if sc > best_s1:
        best_s1, best_w = sc, w

oof_t1_blend = best_w * my_oof_t1 + (1 - best_w) * col_oof_t1_full
test_t1_blend = best_w * my_test_t1 + (1 - best_w) * col_test_t1_full
print(f"\nTask 1 - COMBINAT (w_tau={best_w:.2f}) -> {best_s1:.4f}")

# Calibrare ușoară și conservatoare pe cele 3 clase Clarity
w3 = np.ones(3)
best_s1_cal = best_s1
for _ in range(3):
    for c in range(3):
        for mult in np.linspace(0.75, 1.45, 15):
            tmp = w3.copy()
            tmp[c] = mult
            sc = f1_score(y_clarity, np.argmax(oof_t1_blend * tmp, axis=1), average="macro")
            if sc > best_s1_cal:
                best_s1_cal, w3 = sc, tmp

w3_safe = 0.85 * w3 + 0.15 * np.ones(3)
final_test_t1 = test_t1_blend * w3_safe
s1_final_cv = f1_score(y_clarity, np.argmax(oof_t1_blend * w3_safe, axis=1), average="macro")
print(f"Task 1 - COMBINAT + CALIBRAT   -> {s1_final_cv:.4f}")

# ================= 6. TASK 2 (ALINIAT + CALIBRARE CONSERVATOARE) =================
oof_t2_blend = col_oof_t2
test_t2_blend = col_test_t2_aligned

w9 = np.ones(9)
best_s2_cal = f1_score(y_evasion, np.argmax(oof_t2_blend, axis=1), average="macro")
for _ in range(3):
    for c in range(9):
        for mult in np.linspace(0.75, 1.45, 15):
            tmp = w9.copy()
            tmp[c] = mult
            sc = f1_score(y_evasion, np.argmax(oof_t2_blend * tmp, axis=1), average="macro")
            if sc > best_s2_cal:
                best_s2_cal, w9 = sc, tmp

# Păstrăm o temperare mai prudentă (70% calibrat + 30% 1.0) pe cele 9 clase pentru siguranță maximă pe test
w9_safe = 0.70 * w9 + 0.30 * np.ones(9)
final_test_t2 = test_t2_blend * w9_safe
s2_final_cv = f1_score(y_evasion, np.argmax(oof_t2_blend * w9_safe, axis=1), average="macro")
print(f"Task 2 - FINAL CALIBRAT        -> {s2_final_cv:.4f}")
print(f"SCOR MEDIU CV ESTIMAT          -> {0.5 * (s1_final_cv + s2_final_cv):.4f}")

# ================= 7. GENERARE SUBMISIE FINALĂ =================
sub1 = pd.DataFrame({
    "subtaskID": 1,
    "datapointID": test_df["datapointID"],
    "answer": [CLARITY_LABELS[i] for i in np.argmax(final_test_t1, axis=1)],
})

sub2 = pd.DataFrame({
    "subtaskID": 2,
    "datapointID": test_df["datapointID"],
    "answer": [id2label[i] for i in np.argmax(final_test_t2, axis=1)],
})

submission = pd.concat([sub1, sub2], ignore_index=True)
submission.to_csv("submission_polero_team_blend_aligned.csv", index=False)

print("\nSalvat submission_polero_team_blend_aligned.csv | Shape:", submission.shape, "| Nulls:", submission.isna().sum().sum())
print("\nDistribuție Subtask 1:\n", submission[submission["subtaskID"] == 1]["answer"].value_counts())
print("\nDistribuție Subtask 2:\n", submission[submission["subtaskID"] == 2]["answer"].value_counts())