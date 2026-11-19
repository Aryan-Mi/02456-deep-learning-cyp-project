import sys
import random
import torch
import numpy as np
from best_models import DEVICE, ROOT, all_models
sys.path.insert(0, str(ROOT))
from finetune import metrics, train_torch
from models.gbm import LightGBMModel
from prepare_data import REG
import pandas as pd
from pathlib import Path

data_dir = "data"

def main():
    train_df = pd.read_csv("data/regression/train.csv")
    val_df = pd.read_csv("data/regression/val.csv")
    test_df = pd.read_csv("data/regression/test.csv")

    output_path = "runs/regression"

    train_df = train_df[train_df[REG].notna().any(axis=1)].reset_index(drop=True)
    val_df = val_df[val_df[REG].notna().any(axis=1)].reset_index(drop=True)

    print("Device:", DEVICE)

    val_pred={}
    test_pred ={}

    for model,params in all_models.items():
        print("Training model :", model)

        # as in finetune
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        if params.model.type =="lightgbm":
            model_gbm = LightGBMModel(params.model,Path(data_dir),42 )
            model_gbm.fit(train_df, val_df, REG, "regression", Path(output_path))
            predict = model_gbm.predict
        else:
            pass
        val_pred[model] = predict(val_df)
        test_pred[model] = predict(test_df)
    print(val_pred)


if __name__=="__main__":
    main()