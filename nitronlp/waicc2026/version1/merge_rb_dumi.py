import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ================= 1. ÎNCĂRCĂM DATELE & DEFINIM ETICHETELE =================
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

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
    p_clear = probs9[:, [0]].sum(axis=1)                 # Explicit
    p_ambiv = probs9[:, [1, 2, 3, 4, 5]].sum(axis=1)     # Implicit .. Deflection
    p_nonrep = probs9[:, [6, 7, 8]].sum(axis=1)          # Declining .. Clarification
    return np.stack([p_clear, p_ambiv, p_nonrep], axis=1)

# ================= 2. ÎNCĂRCĂM MATRICILE .NPY =================
oof_dumi = np.load("oof_robert9.npy")       # Dumitrescu (~0.50)
test_dumi = np.load("test_robert9.npy")

oof_rb = np.load("oof_rb9.npy")             # ReaderBench (~0.48) - verifică numele exact al fișierului tău!
test_rb = np.load("test_rb9.npy")

# Normalizare de siguranță dacă vreun fișier a fost salvat fără / 5.0
if oof_dumi[0].sum() > 1.5: oof_dumi /= 5.0
if test_dumi[0].sum() > 1.5: test_dumi /= 5.0
if oof_rb[0].sum() > 1.5: oof_rb /= 5.0
if test_rb[0].sum() > 1.5: test_rb /= 5.0

def get_score(probs9):
    s1 = f1_score(y_clarity, np.argmax(probs9_to_probs3(probs9), axis=1), average="macro")
    s2 = f1_score(y_evasion, np.argmax(probs9, axis=1), average="macro")
    return s1, s2, 0.5 * (s1 + s2)

print("Scor Dumitrescu :", [round(x, 4) for x in get_score(oof_dumi)])
print("Scor ReaderBench:", [round(x, 4) for x in get_score(oof_rb)])

# ================= 3. CĂUTĂM PONDEREA OPTIMĂ (ALPHA) =================
best_alpha, best_raw = 0.55, 0.0
for alpha in np.linspace(0, 1, 21):
    _, _, sc = get_score(alpha * oof_dumi + (1 - alpha) * oof_rb)
    if sc > best_raw:
        best_raw, best_alpha = sc, alpha

oof_ens = best_alpha * oof_dumi + (1 - best_alpha) * oof_rb
test_ens = best_alpha * test_dumi + (1 - best_alpha) * test_rb
print(f"Ensemble brut (Dumi={best_alpha:.2f}, RB={1-best_alpha:.2f}) -> {[round(x, 4) for x in get_score(oof_ens)]}")

# ================= 4. CALIBRARE MULTIPLICATORI PE 9 CLASE =================
weights = np.ones(9)
best_cal = best_raw
for _ in range(2):
    for c in range(9):
        for mult in np.linspace(0.6, 2.2, 17):
            temp_w = weights.copy()
            temp_w[c] = mult
            _, _, sc = get_score(oof_ens * temp_w)
            if sc > best_cal:
                best_cal, weights = sc, temp_w

s1, s2, mean_s = get_score(oof_ens * weights)
print(f"Ensemble calibrat -> Sub1: {s1:.4f} | Sub2: {s2:.4f} | Mediu: {mean_s:.4f}")

# ================= 5. GENERARE SUBMISIE FINALĂ =================
final_test9 = test_ens * weights
final_test3 = probs9_to_probs3(final_test9)

sub1 = pd.DataFrame({
    "subtaskID": 1,
    "datapointID": test_df["datapointID"],
    "answer": [CLARITY_LABELS[i] for i in np.argmax(final_test3, axis=1)],
})

sub2 = pd.DataFrame({
    "subtaskID": 2,
    "datapointID": test_df["datapointID"],
    "answer": [id2label[i] for i in np.argmax(final_test9, axis=1)],
})

submission = pd.concat([sub1, sub2], ignore_index=True)
submission.to_csv("submission_polero_merged.csv", index=False)

print("Salvat submission_polero_merged.csv | Shape:", submission.shape, "| Nulls:", submission.isna().sum().sum())
print(submission.head())