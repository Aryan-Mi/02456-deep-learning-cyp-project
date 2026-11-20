import pandas as pd

seed_runs = [
    "task3/e2e_s42",
    "task3/e2e_s43",
    "task3/e2e_s44",
    "task3/warmup5",
    "task3/warmup5_s43",
    "task3/warmup5_s44",
]

table = pd.read_csv("results_summary.csv")

table = table[table["experiment"].isin(seed_runs)].copy()

config_columns = ["task", "model", "frozen", "warmup_epochs"]

grouped = table.groupby(config_columns)["score"]

summary = grouped.agg(["count", "mean", "std", "min", "max"])
summary = summary.reset_index()

regression = summary[summary["task"] == "regression"].sort_values("mean")
classification = summary[summary["task"] == "classification"].sort_values("mean", ascending = False)

summary = pd.concat([regression, classification])
summary.to_csv("results_summary_aggregated.csv", index = False)


print("Regression - remember: lower is better")
print(regression.to_string(index = False))
print()
print("Classification")
print(classification.to_string(index = False))
print()
print("Saved in results_summary_aggregated.csv")