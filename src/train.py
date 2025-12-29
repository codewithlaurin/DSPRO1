# Train final model using all training data with optional test set evaluation
import argparse
import json
import os
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets.folder import ImageFolder
from tqdm import tqdm

import wandb
from dataset import DATA_TRANSFORMS, get_test_dataset, get_train_dataset
from utils import set_seed


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(args):
    set_seed(42)
    device = get_device()
    print("Device:", device)

    with wandb.init(
        entity="mara-eckart-hochschule-luzern",
        project="plant-health-classification",
        job_type="final-training",
        config=args,
    ) as run:
        train_ds = get_train_dataset(DATA_TRANSFORMS["train"])
        test_ds = get_test_dataset()

        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        print(f"SAMPLES | TRAIN: {len(train_ds)} | TEST: {len(test_ds)}")
        print("Classes:", train_ds.classes)

        model = init_model(train_ds).to(device)

        criterion = nn.CrossEntropyLoss()

        if args.optimizer == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        else:
            optim = torch.optim.SGD(
                model.parameters(), lr=args.learning_rate, momentum=args.momentum
            )

        model.train()
        for epoch in tqdm(range(args.epochs), "Epoch", leave=False):
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            total_preds = []
            total_targets = []

            for x, y in tqdm(train_dl, "Train", leave=False):
                x = x.to(device)
                y = y.to(device)

                optim.zero_grad(set_to_none=True)

                outputs = model(x)

                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, y)

                loss.backward()
                optim.step()

                total_samples += x.size(0)
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y.data).item()

                total_preds.extend(preds.cpu().tolist())
                total_targets.extend(y.cpu().tolist())

            train_loss = running_loss / total_samples
            train_acc = running_corrects / total_samples
            train_f1 = f1_score(total_targets, total_preds, average="macro")

            tqdm.write(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f}"
            )

            run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                }
            )

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        class_names_path = os.path.join(out, "class_names.json")
        model_path = os.path.join(out, "plant_disease_model.pt")

        with open(class_names_path, "w") as f:
            json.dump(train_ds.classes, f)
        torch.save(model.state_dict(), model_path)

        artifact = wandb.Artifact(name=f"plant-disease-model-{run.id}", type="model")
        artifact.add_file(model_path)
        artifact.add_file(class_names_path)
        run.log_artifact(artifact)

        if not args.test:
            return

        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            total_preds = []
            total_targets = []

            for x, y in tqdm(test_dl, "Test", leave=False):
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)

                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, y)

                total_samples += x.size(0)
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y.data).item()

                total_preds.extend(preds.cpu().tolist())
                total_targets.extend(y.cpu().tolist())

            test_loss = running_loss / total_samples
            test_acc = running_corrects / total_samples
            test_f1 = f1_score(total_targets, total_preds, average="macro")
        print(
            f"TEST | test_loss={test_loss:.4f} test_acc={test_acc:.3f} test_f1={test_f1:.3f}"
        )

        run.summary["test_loss"] = test_loss
        run.summary["test_acc"] = test_acc
        run.summary["test_f1"] = test_f1


def init_model(dataset: ImageFolder):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze layer params
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder with train/ and test/")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--optimizer", default="sgd")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    main(args)
