import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sympy.logic.boolalg import true
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision import models
from torchvision.datasets.folder import ImageFolder
from tqdm import tqdm

from dataset import DATA_TRANSFORMS, PROJECT_ROOT, get_train_dataset

K_FOLDS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_EPOCHS = 25

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def evaluate_params():
    kfold = StratifiedKFold(K_FOLDS, shuffle=True, random_state=42)

    dataset: ImageFolder = get_train_dataset()

    fold_params = []
    fold_scores = []

    print(f"PARAMS: K_FOLDS {K_FOLDS} | BATCH_SIZE {BATCH_SIZE} | LEARNING_RATE {LEARNING_RATE} | MOMENTUM {MOMENTUM} | NUM_EPOCHS {NUM_EPOCHS}")
    print(f"Using {device} device")
    print()

    for fold, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(dataset.samples, dataset.targets)), "KFold", leave=False
    ):
        train_folder = get_train_dataset(DATA_TRANSFORMS["train"])
        val_folder = get_train_dataset(DATA_TRANSFORMS["val"])

        fold_train = Subset(train_folder, train_idx)
        fold_val = Subset(val_folder, val_idx)

        train_loader = DataLoader(fold_train, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(fold_val, batch_size=BATCH_SIZE, shuffle=False)

        model = init_model(dataset).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), LEARNING_RATE, MOMENTUM)

        model_params_path, best_val_metrics = train_fold(
            fold, model, criterion, optimizer, NUM_EPOCHS, train_loader, val_loader
        )

        fold_params.append(model_params_path)
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

    return summary, fold_params


def init_model(dataset: ImageFolder):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze layer params
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    return model


def train_fold(fold, model, criterion, optimizer, num_epochs, train_loader, val_loader):
    model_params_path = os.path.join(OUTPUT_DIR, f"fold{fold}_model_params.pt")

    best_metrics = {"f1": -1.0, "acc": 0.0, "loss": float("inf")}

    for epoch in tqdm(range(num_epochs), "Epoch", leave=False):
        model.train()

        _, train_loss, train_acc = epoch_phase(
            model, criterion, optimizer, train_loader
        )

        print(
            f"Fold {fold:02d} | Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
        )

        model.eval()

        with torch.no_grad():
            val_f1, val_loss, val_acc = epoch_phase(
                model, criterion, optimizer, val_loader, False
            )

        print(
            f"Fold {fold:02d} | Epoch {epoch:02d} | val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )

        if val_f1 > best_metrics["f1"] or (
            val_f1 == best_metrics["f1"] and val_loss < best_metrics["loss"]
        ):
            best_metrics["f1"] = val_f1
            best_metrics["acc"] = val_acc
            best_metrics["loss"] = val_loss
            torch.save(model.state_dict(), model_params_path)

    return model_params_path, best_metrics


def epoch_phase(model, criterion, optimizer, loader, train=True):
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    total_preds = []
    total_targets = []

    for x, y in tqdm(loader, "Train" if train else "Val", leave=False):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(x)

        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, y)

        if train:
            loss.backward()
            optimizer.step()

        total_samples += x.size(0)
        running_loss += loss.item() * x.size(0)
        running_corrects += torch.sum(preds == y.data)

        if not train:
            total_preds.extend(preds.cpu().tolist())
            total_targets.extend(y.cpu().tolist())

    loss = running_loss / total_samples
    acc = running_corrects.double() / total_samples

    if train:
        return 0, loss, acc

    print(total_targets)
    print(total_preds)
    f1 = f1_score(total_targets, total_preds, average="macro")

    return f1, loss, acc


if __name__ == "__main__":
    evaluate_params()
