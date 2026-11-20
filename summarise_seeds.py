import pandas as pd

table = pd.read_csv("results_summary.csv")

config_columns = ["task", "model", "frozen", "warmup_epochs", "embeddings", "descriptors"]

for column in config_columns:
    table[column] = table[column].fillna("-")

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