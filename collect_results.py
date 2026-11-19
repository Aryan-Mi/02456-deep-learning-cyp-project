import json
import os 
from pathlib import Path
import pandas as pd

rows = []
per_target = []

config_files = []
for folder_name, subfolders, filenames in os.walk("runs"):
    if "config.json" in filenames:
        config_files.append(Path(folder_name) / "config.json")

config_files.sort()

for config_file in config_files:
    folder = config_file.parent
    results_file = folder / "validation_metrics.json"
    if not results_file.exists():
        print(f"Skipping {folder} because {results_file} does not exist")
        continue

    config = json.loads(config_file.read_text())
    results = json.loads(results_file.read_text())

    task = config["task"]
    if task == "regression":
        score_name = "macro_ST-RAE"
    else:
        score_name = "macro_MCC"

    best_epoch = None
    best_score = None
    history_file = folder / "history.jsonl"

    if history_file.exists():
        history = []

        for line in history_file.open():
            if line.strip():
                history.append(json.loads(line))

        for epoch_record in history:
            score = epoch_record[score_name]

            if best_score is None:
                is_better = True
            elif task == "regression":
                is_better = score < best_score
            else:
                is_better = score > best_score

            if is_better:
                best_score = score
                best_epoch = epoch_record["epoch"]

    model = config["model"]
    train = config["train"]

    featurizers = model.get("featurizers")
    if featurizers is None:
        desriptors = " "
    else:
        desriptors = " ".join(featurizers)

    row = {
        "task": task,
        "experiment": config.get("experiment_name"),
        "seed": config.get("seed"),
        "model": model.get("name"),

        "frozen": model.get("freeze_encoder"),
        "warmup_epochs": train.get("head_only_epochs"),

        "embeddings": model.get("use_embeddings"),
        "descriptors": desriptors,
        "epochs": train.get("epochs"),
        "best_epoch": best_epoch,
        "score": results[score_name],
    }
    rows.append(row)

    for target, target_scores in results.items():
        if target.startswith("macro"):
            continue

        target_row = {
            "task": task,
            "experiment": config.get("experiment_name"),
            "seed": config.get("seed"),
            "target": target,
        }
        for metric_name, metric_value in target_scores.items():
            target_row[metric_name] = metric_value
        per_target.append(target_row)

table = pd.DataFrame(rows)
table = table.sort_values(["task", "experiment", "seed"])
table.to_csv("results_summary.csv", index=False)

target_table = pd.DataFrame(per_target)
target_table.to_csv("results_per_target.csv", index=False)

print()
print(table.to_string(index=False))
print()
print("Done -", len(table), "runs collected.")
print("Saved results_summary.csv and results_per_target.csv")