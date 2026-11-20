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
            print("Chemeleon")
            predict = train_torch(params, train_df, val_df, REG, "regression", Path(data_dir), Path(output_path))
        val_pred[model] = predict(val_df)
        test_pred[model] = predict(test_df)
        del predict
        torch.cuda.empty_cache()

    with open("ensembles/results.txt", "w") as f:
        print("Regression ------------------------------", file = f)
        for split, dataset, predictions in [("validation", val_df, val_pred), ("test", test_df, test_pred)]:
            avg_pred = np.mean(list(predictions.values()), axis=0)
            print(f"Split: {split} ------------------------", file = f)
            for model in all_models:
                print(f"model: {model}", file = f)
                model_score = metrics(dataset, predictions[model], REG, "regression")["macro_ST-RAE"]
                print(f"ST RAE model : {model_score}", file=f)
            ensamble_score = metrics(dataset, avg_pred, REG,"regression")["macro_ST-RAE"]
            print(f"Ensamble score : {ensamble_score}", file=f)
        


if __name__=="__main__":
    main()