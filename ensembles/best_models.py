from pathlib import Path
import torch
from omegaconf import OmegaConf

'''
File with the best model configurations from parts 1-3 (to create an ensamble of the best models)
'''

ROOT = Path(__file__).resolve().parent.parent 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHEMELEON = OmegaConf.create(
    {
        "model": {
            "type": "chemeleon",
            "checkpoint": str(ROOT/"checkpoints"/"chemeleon_mp.pt"),
            "freeze_encoder": False,
            "readout_lr": 1e-3,
            "encoder_lr": 1e-4,
        },
        "readout": {
            "type": "mlp",
            "hidden_dim": 512,
            "num_layers": 3,
            "dropout": 0.3,
        }, # to change to epochs 20 and head only 5 before running on GPU
        "train": {
            "epochs": 2,
            "batch_size": 32,
            "head_only_epochs": 1,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "reduce_lr": False,
        },
        "device": DEVICE,
    }
)

# to be changed after Anyas best results
ATOM_JEPA = OmegaConf.create(
    {
        "model": {
            "type": "atom_jepa",
            "checkpoint": "molecules",  
            "precision": "fp32",
            "freeze_encoder": False,
            "encoder_lr": 1.0e-5,
            "readout_lr": 1.0e-3,
        },
        "readout": {"type": "mlp"},
        "train": {
            "epochs": 20,
            "batch_size": 8,
            "weight_decay": 1.0e-4,
            "grad_clip": 1.0,
            "head_only_epochs": 0,
            "reduce_lr": False,
        },
        "device": DEVICE,
    }
)

LIGHTGBM = OmegaConf.create(
    {
        "model": {
            "type": "lightgbm",
            "featurizers": ["morgan", "rdkit"],
            "morgan": {"radius": 2, "n_bits": 1024, "use_counts": True},
            "feature_workers": 4,
            "early_stopping_rounds": 200,
            "params": { 
                "n_estimators": 5000,
                "learning_rate": 0.03,
                "num_leaves": 15,
                "min_child_samples": 20,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.3,
                "reg_lambda": 1.0,
                "n_jobs": 4,
            },
        },
        "device": DEVICE,
    }
)



all_models = {
    "chemeleon": CHEMELEON,
    # "atom_jepa": ATOM_JEPA,
    "lightgbm": LIGHTGBM,
}

