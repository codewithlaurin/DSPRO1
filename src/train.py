import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import ImageFolder
from tqdm import tqdm

from dataset import DATA_TRANSFORMS, PROJECT_ROOT, get_train_dataset

K_FOLDS = 10
BATCH_SIZE = 64

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def evaluate_params():
    kfold = StratifiedKFold(K_FOLDS, shuffle=True, random_state=42)

    dataset: ImageFolder = get_train_dataset()

    fold_ckpts = []
    fold_scores = []

    print("training folds...")
    for fold, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(dataset.samples, dataset.targets))
    ):
        train_folder = get_train_dataset(DATA_TRANSFORMS["train"])
        val_folder = get_train_dataset(DATA_TRANSFORMS["val"])

        fold_train = Subset(train_folder, train_idx)
        fold_val = Subset(val_folder, val_idx)

        train_loader = DataLoader(fold_train, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(fold_val, batch_size=BATCH_SIZE, shuffle=False)

        # train model using train_loader/val_loader -> returns model & dict of best metrics

        best_state, best_val_metrics = train_fold(train_loader, val_loader)

        ckpt_path = os.path.join(OUTPUT_DIR, f"models/model_fold{fold}.pth")
        torch.save(best_state, ckpt_path)
        fold_ckpts.append(ckpt_path)

        fold_scores.append(best_val_metrics)

    keys = fold_scores[0].keys()
    summary = {
        k: (
            np.mean([fs[k] for fs in fold_scores]),
            np.std([fs[k] for fs in fold_scores], ddof=1),
        )
        for k in keys
    }

    print("--- Cross-validation metrics summary ---")
    for k, (m, s) in summary.items():
        print(f"{k}: {m:.4f} ± {s:.4f}")

    return summary, fold_ckpts


def train_fold(train_loader, val_loader):
    # train model -> returns model & dict of best metrics
    pass
