import argparse, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

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

    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, running = 0, 0, 0.0
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
        print(f"Epoch {epoch:02d} | train_loss={running/total:.4f} train_acc={correct/total:.3f}")

    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"TEST acc = {correct/total:.3f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out/"resnet18_binary.pt")
    print("Saved:", out/"resnet18_binary.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder with train/ and test/")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    main(args)
