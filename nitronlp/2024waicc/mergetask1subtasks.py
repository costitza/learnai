import pandas as pd

# 1. Load both files
df_t1 = pd.read_csv("submission_task1_1.csv")
df_t2 = pd.read_csv("submission_task1_2.csv")

# 2. Extract only subtaskID == 1 from the first file and subtaskID == 2 from the second
sub1 = df_t1[df_t1["subtaskID"] == 1][["subtaskID", "datapointID", "answer"]].copy()
sub2 = df_t2[df_t2["subtaskID"] == 2][["subtaskID", "datapointID", "answer"]].copy()

# 3. Concatenate and save without index
final_sub = pd.concat([sub1, sub2], ignore_index=True)
final_sub.to_csv("submission_final.csv", index=False)

# 4. Sanity checks before uploading
print("Shape:", final_sub.shape)
print("Subtask counts:\n", final_sub["subtaskID"].value_counts())
print("Null values:", final_sub.isna().sum().sum())
print(final_sub.head(3))
print(final_sub.tail(3))