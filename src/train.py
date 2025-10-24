import argparse, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score
import wandb 

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class BinaryMapFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self._bin = {idx: (1 if "healthy" in name.lower() else 0)
                    for name, idx in self.class_to_idx.items()}
    def __getitem__(self, i):
        img, y = super().__getitem__(i)
        return img, self._bin[y]

def main(args):
    wandb.init(
        project="plant-health-classification",  
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "SGD",
            "architecture": "ResNet18",
        },
    )

    device = get_device()
    print("Device:", device)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = BinaryMapFolder(Path(args.data_root)/"train", transform=tf)
    test_ds  = BinaryMapFolder(Path(args.data_root)/"test",  transform=tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print("Classes:", train_ds.classes)


    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    wandb.watch(model, criterion, log="all", log_freq=100)

    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, running = 0, 0, 0.0
        train_preds, train_targets = [], []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            train_preds.extend(logits.argmax(1).detach().cpu().tolist())
            train_targets.extend(y.detach().cpu().tolist())
        train_f1 = f1_score(train_targets, train_preds)
        train_loss = running / total
        train_acc = correct / total
        #not really necessary
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "train/f1": train_f1,
        })

    model.eval()
    total, correct = 0, 0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            test_preds.extend(pred.cpu().tolist())
            test_targets.extend(y.cpu().tolist())
    test_f1 = f1_score(test_targets, test_preds)
    test_acc = correct / total
    print(f"TEST acc = {test_acc:.3f} TEST f1 = {test_f1:.3f}")

    wandb.log({
        "test/acc": test_acc,
        "test/f1": test_f1,
    })

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model_path = out / "resnet18_binary.pt"
    torch.save(model.state_dict(), out/"resnet18_binary.pt")
    print("Saved:", out/"resnet18_binary.pt")
    wandb.save(str(model_path))

    wandb.finish()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder with train/ and test/")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    main(args)
