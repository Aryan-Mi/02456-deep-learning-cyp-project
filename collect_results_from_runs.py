import json
from pathlib import Path
import pandas as pd

rows, per_target = [], []
for cfg_path in sorted(Path("runs2").rglob("config.json")):
    run = cfg_path.parent
    cfg = json.loads(cfg_path.read_text())
    task = cfg["task"]
    macro = "macro_ST-RAE" if cfg["task"] == "regression" else "macro_MCC"
    lower = task == "regression"
    overrides = (run / ".hydra" / "overrides.yaml").read_text().splitlines()
    overrides = [o.lstrip("- ").strip() for o in overrides if o.strip()]
    vm = json.loads((run / "validation_metrics.json").read_text())
    hist = best_epoch = final_score = None
    hp = run / "history.jsonl"
    if hp.exists():
        hist = pd.DataFrame([json.loads(l) for l in hp.open() if l.strip()])
        hist = hist.drop_duplicates("epoch", keep="last").sort_values("epoch")
        idx = hist[macro].idxmin() if lower else hist[macro].idxmax()
        best_epoch = int(hist.loc[idx, "epoch"])
        final_score = float(hist[macro].iloc[-1])
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
        "best_epoch": best_epoch,
        "epochs_run": None if hist is None else len(hist),
        "final_score": final_score,
        "train_loss_first": None if hist is None else float(hist.train_loss.iloc[0]),
        "train_loss_last":  None if hist is None else float(hist.train_loss.iloc[-1]),
        "overrides": " ".join(overrides),
    })

    for target, m in vm.items():
        if isinstance(m, dict):            # pomija macro_*, które jest liczbą
            per_target.append({"task": task, "exp": cfg["experiment_name"],
                               "seed": cfg["seed"], "target": target, **m})

pd.DataFrame(rows).to_csv("notebooks/results/chemeleon/all_runs2.csv", index=False)
pd.DataFrame(per_target).to_csv("notebooks/results/chemeleon/per_target2.csv", index=False)
print(len(rows), "runs ->", len(per_target), "per-target rows")
