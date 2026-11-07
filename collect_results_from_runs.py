import json
from pathlib import Path
import pandas as pd

rows = []
for cfg_path in sorted(Path("runs").rglob("config.json")):
    run = cfg_path.parent
    cfg = json.loads(cfg_path.read_text())
    macro = "macro_ST-RAE" if cfg["task"] == "regression" else "macro_MCC"

    overrides = (run / ".hydra" / "overrides.yaml").read_text().splitlines()
    overrides = [o.lstrip("- ").strip() for o in overrides if o.strip()]

    rows.append({
        "task": cfg["task"],
        "exp": cfg["experiment_name"],
        "seed": cfg["seed"],
        "freeze_encoder": cfg["model"]["freeze_encoder"],
        "encoder_lr": cfg["model"]["encoder_lr"],
        "head_lr": cfg["model"]["readout_lr"],
        "readout": cfg["readout"]["type"],
        "hidden_dim": cfg["readout"].get("hidden_dim"),
        "num_layers": cfg["readout"].get("num_layers"),
        "dropout": cfg["readout"].get("dropout"),
        "epochs": cfg["train"]["epochs"],
        "batch_size": cfg["train"]["batch_size"],
        "weight_decay": cfg["train"]["weight_decay"],
        "grad_clip": cfg["train"]["grad_clip"],
        "head_only_epochs": cfg["train"].get("head_only_epochs", 0),
        "reduce_lr": cfg["train"].get("reduce_lr", False),
        "lr_factor": cfg["train"].get("lr_factor"),
        "lr_patience": cfg["train"].get("lr_patience"),
        "val_score": json.loads((run / "validation_metrics.json").read_text())[macro],
        "baseline_score": json.loads((run / "baseline_metrics.json").read_text())[macro],
        "overrides": " ".join(overrides),
    })

pd.DataFrame(rows).to_csv("notebooks/results/chemeleon/all_runs.csv", index=False)
print(len(rows), "runs-> all_runs.csv")