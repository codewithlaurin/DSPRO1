# Training loop with hold out validation set for W&B sweep
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data.dataloader import DataLoader
from torchvision import models
from tqdm import tqdm

import wandb
from dataset import DATA_TRANSFORMS, PROJECT_ROOT, get_train_dataset, get_val_dataset
from utils import set_seed

K_FOLDS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_EPOCHS = 150
WEIGHT_DECAY = 0.00001

OUTPUT_DIR = Path(PROJECT_ROOT) / "output"
MODEL_PARAMS_PATH = OUTPUT_DIR / "best_model_params.pt"


def setup_env():
    set_seed(42)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = setup_env()


def init_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze layer params
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_optimizer(params, config, lr):
    if config.optimizer == "adam":
        return optim.Adam(params, lr, weight_decay=config.weight_decay)
    else:
        return optim.SGD(params, lr, config.momentum, weight_decay=config.weight_decay)


def evaluate_params():
    # manual config -> remove for sweep
    config_defaults = {
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "adam",
        "architecture": "ResNet18",
    }

    with wandb.init(
        entity="mara-eckart-hochschule-luzern",
        project="plant-health-classification",
        config=config_defaults,
    ) as run:
        config = run.config
        print("PARAMS", end="")
        for key, value in config.items():
            print(" | " + str(key) + ": " + str(value), end="")
        print()
        print(f"Using {DEVICE} device")

        train_ds = get_train_dataset(DATA_TRANSFORMS["train"])
        val_ds = get_val_dataset()

        print(f"SAMPLES | {len(train_ds)}")

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        model = init_model(len(train_ds.classes)).to(DEVICE)

        tqdm.write("--- Phase 1: Training Head ---")

        optimizer = get_optimizer(model.parameters(), config, config.learning_rate)

        # Train head
        train(
            run,
            model,
            optimizer,
            5,
            train_loader,
            val_loader,
        )

        tqdm.write("--- Phase 2: Fine Tuning ---")

        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True

        optimizer = get_optimizer(
            model.parameters(), config, config.learning_rate / 100
        )

        # Fine tune
        train(
            run,
            model,
            optimizer,
            config.epochs,
            train_loader,
            val_loader,
        )

        model.load_state_dict(torch.load(MODEL_PARAMS_PATH, weights_only=True))

        model.eval()
        total_preds, total_targets = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, "Conf matrix", leave=False):
                x = x.to(DEVICE)
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                total_preds.extend(preds.cpu().tolist())
                total_targets.extend(y.tolist())

        run.log(
            {
                f"conf_mat": wandb.plot.confusion_matrix(
                    None, total_targets, total_preds, train_ds.classes
                )
            }
        )

        with open(OUTPUT_DIR / "class_names.json", "w") as f:
            json.dump(train_ds.classes, f)

        run.save(str(OUTPUT_DIR / "*"))


def train(run, model, optimizer, num_epochs, train_loader, val_loader, patience=7):
    best_f1 = -1
    best_loss = float("inf")

    early_stop_count = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3)

    for epoch in tqdm(range(num_epochs), "Epoch", leave=False):
        _, train_loss, train_acc = run_epoch_phase(model, optimizer, train_loader)

        tqdm.write(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
        )

        with torch.no_grad():
            val_f1, val_loss, val_acc = run_epoch_phase(
                model, optimizer, val_loader, False
            )

        scheduler.step(val_f1)

        tqdm.write(
            f"Epoch {epoch:02d} | val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )

        run.log(
            {
                f"epoch": epoch,
                f"train_loss": train_loss,
                f"train_acc": train_acc,
                f"val_loss": val_loss,
                f"val_acc": val_acc,
                f"val_f1": val_f1,
            },
        )

        if val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_loss):
            best_f1 = val_f1
            best_loss = val_loss
            run.summary[f"best_loss"] = val_loss
            run.summary[f"best_acc"] = val_acc
            run.summary[f"best_f1"] = val_f1
            torch.save(model.state_dict(), MODEL_PARAMS_PATH)
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            break


def run_epoch_phase(model, optimizer, loader, train=True):
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    total_preds = []
    total_targets = []

    model.train() if train else model.eval()

    for x, y in tqdm(loader, "Train" if train else "Val", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        if train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(x)

        _, preds = torch.max(outputs, 1)

        loss = F.cross_entropy(outputs, y)

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


if __name__ == "__main__":
    evaluate_params()
