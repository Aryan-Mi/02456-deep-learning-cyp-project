# %%
import pandas as pd
from scipy import stats

freezing = {
    "frozen": ["task3/frozen_s42", "task3/frozen_s43", "task3/frozen_s44"],
    "end-to-end": ["task3/e2e_s42", "task3/e2e_s43", "task3/e2e_s44"],
    "warmup": ["task3/warmup5", "task3/warmup5_s43", "task3/warmup5_s44"]
}

readout = {
    "descriptors": ["task3/lgbm_descriptors", "task3/lgbm_descriptors_s43", "task3/lgbm_descriptors_s44"],
    "embeddings": ["task3/lgbm_embeddings", "task3/lgbm_embeddings_s43", "task3/lgbm_embeddings_s44"],
    "both": ["task3/lgbm_both", "task3/lgbm_both_s43", "task3/lgbm_both_s44"]
}

readout_architecture = {
    "MLP head": ["task3/frozen_s42", "task3/frozen_s43", "task3/frozen_s44"],
    "LightGBM": ["task3/lgbm_embeddings", "task3/lgbm_embeddings_s43", "task3/lgbm_embeddings_s44"]
}

all_runs = pd.read_csv("results_summary.csv")

def label_runs(groups):
    kept = []
    labels = []

    for index, row in all_runs.iterrows(): 
        for setup_name in groups:
            if row["experiment"] in groups[setup_name]:
                kept.append(index)
                labels.append(setup_name)

    table = all_runs.loc[kept].copy()
    table["setup"] = labels

    return table

def summarise(direction_name, groups):
    table = label_runs(groups)

    summary = table.groupby(["task", "setup"])["score"].agg(["count", "mean", "std", "min", "max"])
    summary = summary.reset_index()

    regression = summary[summary["task"] == "regression"].sort_values("mean")
    classification = summary[summary["task"] == "classification"].sort_values("mean", ascending = False)

    print(direction_name)
    print("Regression")
    print(regression.to_string(index = False))
    print("Classification")
    print(classification.to_string(index = False))

    summary["direction"] = direction_name
    return summary

def anova(direction_name, groups):
    table = label_runs(groups)
    anova_rows = []

    for task_name in ["regression", "classification"]:
        task_table = table[table["task"] == task_name]

        score_groups = []
        for setup_name in table["setup"].unique():
            score_groups.append(task_table[task_table["setup"] == setup_name]["score"])

        result = stats.f_oneway(*score_groups)
        anova_rows.append({"direction": direction_name, "task": task_name, "setups": len(score_groups), "p": result.pvalue})

        print(direction_name, "-", task_name, "- ANOVA across", len(score_groups), "setups, p =", result.pvalue)

    return pd.DataFrame(anova_rows)

summaries = []
anovas = []

summaries.append(summarise("Freezing Strategies", freezing))
anovas.append(anova("Freezing Strategies", freezing))

summaries.append(summarise("Readout Features", readout))
anovas.append(anova("Readout Features", readout))

summaries.append(summarise("Readout Architecture", readout_architecture))
anovas.append(anova("Readout Architecture", readout_architecture))

pd.concat(summaries).to_csv("summaries.csv", index = False)
pd.concat(anovas).to_csv("anovas.csv", index = False)

print("Saved summaries.csv and anovas.csv")

# %%

def pairwise_tests(direction_name, groups, task_name):
    table = label_runs(groups)
    task_table = table[table["task"] == task_name]

    setup_names = list(groups)

    pairs = []
    for first_position in range(len(setup_names)):
        for second_position in range(first_position + 1, len(setup_names)):
            pairs.append([setup_names[first_position], setup_names[second_position]])

    corrected_threshold = 0.05 / len(pairs)

    if task_name == "regression":
        lower_is_better = True
    else:
        lower_is_better = False

    pairwise_rows = []

    for pair in pairs:
        first_setup = pair[0]
        second_setup = pair[1]

        first_scores = task_table[task_table["setup"] == first_setup]["score"]
        second_scores = task_table[task_table["setup"] == second_setup]["score"]

        result = stats.f_oneway(first_scores, second_scores)

        if result.pvalue < corrected_threshold:
            verdict = "different"

            if lower_is_better:
                if first_scores.mean() < second_scores.mean():
                    better = first_setup
                else:
                    better = second_setup
            else:
                if first_scores.mean() > second_scores.mean():
                    better = first_setup
                else:
                    better = second_setup
        else:
            verdict = "no evidence"
            better = "-"

        pairwise_rows.append({
            "direction": direction_name,
            "task": task_name,
            "first_setup": first_setup,
            "second_setup": second_setup,
            "first_mean": first_scores.mean(),
            "second_mean": second_scores.mean(),
            "p": result.pvalue,
            "threshold": corrected_threshold,
            "verdict": verdict,
            "better": better
        })
    results = pd.DataFrame(pairwise_rows)

    print(direction_name, "-", task_name, "- a pair counts as different below", round(corrected_threshold, 4))
    print()
    print(results[["first_setup", "second_setup", "first_mean", "second_mean", "p", "verdict", "better"]].to_string(index = False))

    return results
# %%

pairwise_results = []

pairwise_results.append(pairwise_tests("Freezing Strategies", freezing, "regression"))
pairwise_results.append(pairwise_tests("Freezing Strategies", freezing, "classification"))
pairwise_results.append(pairwise_tests("Readout Features", readout, "regression"))

pd.concat(pairwise_results).to_csv("pairwise.csv", index = False)
print("Saved pairwise.csv")

# %%
