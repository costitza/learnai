import pandas as pd

df1 = pd.read_csv("task1.csv")
df2 = pd.read_csv("task2.csv")

if "subtaskID" in df1.columns:
    sub1 = df1[df1["subtaskID"] == 1][["subtaskID", "datapointID", "answer"]].copy()
else:
    sub1 = pd.DataFrame({
        "subtaskID": 1,
        "datapointID": df1["datapointID"],
        "answer": df1["answer"],
    })

if "subtaskID" in df2.columns:
    sub2 = df2[df2["subtaskID"] == 2][["subtaskID", "datapointID", "answer"]].copy()
else:
    sub2 = pd.DataFrame({
        "subtaskID": 2,
        "datapointID": df2["datapointID"],
        "answer": df2["answer"],
    })

submission = pd.concat([sub1, sub2], ignore_index=True)
submission.to_csv("submission_final.csv", index=False)

print("Shape:", submission.shape)
print("Subtask counts:\n", submission["subtaskID"].value_counts())
print("Null values:", submission.isna().sum().sum())
print(submission.head(3))
print(submission.tail(3))