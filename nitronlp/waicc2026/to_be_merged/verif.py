import numpy as np
import pandas as pd

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



my_oof_t1 = np.load("oof_sub1_rb3.npy")              # (N_train, 3) - modelul tău dedicat Task 1
my_test_t1 = np.load("test_sub1_rb3.npy")            # (N_test, 3)

# (Opțional: dacă ai și fișierele tale de pe Task 2, decomentează liniile de mai jos!)
# my_oof_t2 = np.load("oof_readerbench_aug9.npy")
# my_test_t2 = np.load("test_readerbench_aug9.npy")

# --- Implementarea COLEGULUI (actualizează numele exacte ale fișierelor lui!) ---
col_oof_t1 = np.load("oof3_RoBERT-large.npy")             # (N_train, 3)
col_test_t1 = np.load("test3_RoBERT-large.npy")           # (N_test, 3)

col_oof_t2 = np.load("oof_RoBERT-large.npy")             # (N_train, 9)
col_test_t2 = np.load("test_RoBERT-large.npy")           # (N_test, 9)


# 1. Verificăm corelația pe OOF (train) vs TEST
print("Corelație OOF  (train):", np.corrcoef(my_oof_t1[:, 1], col_oof_t1[:, 1])[0, 1])
print("Corelație TEST (raw)  :", np.corrcoef(my_test_t1[:, 1], col_test_t1[:, 1])[0, 1])

# 2. Testăm dacă colegul a sortat test_df după ["url", "question_order", "datapointID"]
sort_idx = test_df.sort_values(["url", "question_order", "datapointID"]).index.values
inv_sort_idx = np.argsort(sort_idx)

print("Corelație dacă colegul a sortat test_df :", np.corrcoef(my_test_t1[:, 1], col_test_t1[inv_sort_idx, 1])[0, 1])
print("Corelație dacă TU ai sortat test_df      :", np.corrcoef(my_test_t1[inv_sort_idx, 1], col_test_t1[:, 1])[0, 1])

# 3. Verificăm și toate permutările celor 3 coloane (în caz că ordinea claselor diferă)
from itertools import permutations
for perm in permutations([0, 1, 2]):
    c = np.corrcoef(my_test_t1[:, 1], col_test_t1[:, perm][:, 1])[0, 1]
    if c > 0.65:
        print(f"Potrivire găsită la permutarea coloanelor {perm}: corelație = {c:.4f}")


sort_dp = test_df.sort_values("datapointID").index.values
inv_dp = np.argsort(sort_dp)
print("Corelație dacă colegul a sortat după datapointID:", np.corrcoef(my_test_t1[:, 1], col_test_t1[inv_dp, 1])[0, 1])