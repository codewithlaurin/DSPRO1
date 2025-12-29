import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision import models
from torchvision.datasets.folder import ImageFolder
from tqdm import tqdm
from utils import set_seed

import wandb
from dataset import DATA_TRANSFORMS, PROJECT_ROOT, get_train_dataset

K_FOLDS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_EPOCHS = 150

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

set_seed(42)

device = get_device()

dataset: ImageFolder = get_train_dataset()


def evaluate_params():
    # manual config -> remove for sweep
    config = {
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "k_folds": K_FOLDS,
        "momentum": MOMENTUM,
        "optimizer": "SGD",
        "architecture": "ResNet18",
    }

    with wandb.init(
        entity="mara-eckart-hochschule-luzern",
        project="plant-health-classification",
        config=config,
    ) as run:
        for fold in range(run.config["k_folds"]):
            run.define_metric(f"fold_{fold}/*", step_metric=f"fold_{fold}/epoch")

        print("PARAMS", end="")
        for key, value in run.config.items():
            print(" | " + str(key) + ": " + str(value), end="")
        print()
        print(f"Using {device} device")

        cross_validation(run, run.config)


def cross_validation(run, config):
    kfold = StratifiedKFold(config["k_folds"], shuffle=True, random_state=42)

    print(f"SAMPLES | {len(dataset)}")

    fold_params = []
    fold_scores = []

    train_folder = get_train_dataset(DATA_TRANSFORMS["train"])
    val_folder = get_train_dataset(DATA_TRANSFORMS["val"])

    for fold, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(dataset.samples, dataset.targets)),
        "KFold",
        config["k_folds"],
        leave=False,
    ):
        fold_train = Subset(train_folder, train_idx)
        fold_val = Subset(val_folder, val_idx)

        train_loader = DataLoader(
            fold_train,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            fold_val,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        model = init_model(dataset).to(device)

        criterion = nn.CrossEntropyLoss()

        if config["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), config["learning_rate"])
        else:
            optimizer = optim.SGD(
                model.parameters(), config["learning_rate"], config["momentum"]
            )

        model_params_path, best_val_metrics = train_fold(
            run,
            fold,
            model,
            criterion,
            optimizer,
            config["epochs"],
            train_loader,
            val_loader,
        )

        fold_params.append(model_params_path)
        fold_scores.append(best_val_metrics)

        tqdm.write(f"--- Fold {fold} metrics ---")
        tqdm.write(
            f"loss={best_val_metrics['loss']:.4f} acc={best_val_metrics['acc']:.3f} f1={best_val_metrics['f1']:.3f}"
        )

        model.load_state_dict(torch.load(model_params_path, weights_only=True))

        run.log({f"fold_{fold}/conf_mat": get_conf_mat(model, val_loader)})

        run.summary[f"fold_{fold}/best_loss"] = best_val_metrics["loss"]
        run.summary[f"fold_{fold}/best_acc"] = best_val_metrics["acc"]
        run.summary[f"fold_{fold}/best_f1"] = best_val_metrics["f1"]

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

    with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
        json.dump(dataset.classes, f)

    run.save(os.path.join(OUTPUT_DIR, "*"))

    run.summary["avg_val_acc"] = summary["acc"][0]
    run.summary["std_val_acc"] = summary["acc"][1]
    run.summary["avg_val_loss"] = summary["loss"][0]
    run.summary["std_val_loss"] = summary["loss"][1]
    run.summary["avg_val_f1"] = summary["f1"][0]
    run.summary["std_val_f1"] = summary["f1"][1]


def init_model(dataset: ImageFolder):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze layer params
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    return model


def train_fold(
    run, fold, model, criterion, optimizer, num_epochs, train_loader, val_loader
):
    model_params_path = os.path.join(OUTPUT_DIR, f"fold{fold}_model_params.pt")

    best_metrics = {"f1": -1.0, "acc": 0.0, "loss": float("inf")}

    early_stop_count = 0

    for epoch in tqdm(range(num_epochs), "Epoch", leave=False):
        model.train()

        _, train_loss, train_acc = run_epoch_phase(
            model, criterion, optimizer, train_loader
        )

        tqdm.write(
            f"Fold {fold:02d} | Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
        )

        model.eval()

        with torch.no_grad():
            val_f1, val_loss, val_acc = run_epoch_phase(
                model, criterion, optimizer, val_loader, False
            )

        tqdm.write(
            f"Fold {fold:02d} | Epoch {epoch:02d} | val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )

        run.log(
            {
                f"fold_{fold}/epoch": epoch,
                f"fold_{fold}/train_loss": train_loss,
                f"fold_{fold}/train_acc": train_acc,
                f"fold_{fold}/val_loss": val_loss,
                f"fold_{fold}/val_acc": val_acc,
                f"fold_{fold}/val_f1": val_f1,
            },
        )

        if val_f1 > best_metrics["f1"] or (
            val_f1 == best_metrics["f1"] and val_loss < best_metrics["loss"]
        ):
            best_metrics["f1"] = val_f1
            best_metrics["acc"] = val_acc
            best_metrics["loss"] = val_loss
            torch.save(model.state_dict(), model_params_path)
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= 15:
                break

    return model_params_path, best_metrics


def run_epoch_phase(model, criterion, optimizer, loader, train=True):
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
        running_corrects += torch.sum(preds == y.data).item()

        if not train:
            total_preds.extend(preds.cpu().tolist())
            total_targets.extend(y.cpu().tolist())

    loss = running_loss / total_samples
    acc = running_corrects / total_samples

    if train:
        return 0, loss, acc

    f1 = f1_score(total_targets, total_preds, average="macro")

    return f1, loss, acc


def get_conf_mat(model, val_loader):
    total_preds = []
    total_targets = []
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(val_loader, "Conf matrix", leave=False):
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            _, preds = torch.max(outputs, 1)

            total_preds.extend(preds.cpu().tolist())
            total_targets.extend(y.cpu().tolist())

    return wandb.plot.confusion_matrix(
        None, total_targets, total_preds, dataset.classes
    )


if __name__ == "__main__":
    evaluate_params()
