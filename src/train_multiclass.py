import argparse, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score
import wandb 
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

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

    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(Path(args.data_root)/"train", transform=tf_train)
    test_ds  = datasets.ImageFolder(Path(args.data_root)/"test",  transform=tf_test)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"#classes: {len(train_ds.classes)}")
    for i, c in enumerate(train_ds.classes):
        print(f"{i:2d}: {c}")


    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
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
            
        train_f1 = f1_score(train_targets, train_preds,average="macro")
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
    
    idx_to_name = {i: name for i, name in enumerate(train_ds.classes)}
    prob_cols = [f"p({name})" for name in train_ds.classes]
    table = wandb.Table(columns=["image", "true_label", "pred_label"])

    max_rows = len(test_preds) 
    rows_added = 0

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1) 
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            test_preds.extend(pred.cpu().tolist())
            test_targets.extend(y.cpu().tolist())


            if rows_added < max_rows:
                for i in range(x.size(0)):
                    if rows_added >= max_rows: break
                    img_tensor = x[i].cpu()

                    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
                    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
                    pil_img = to_pil_image((img_tensor * std + mean).clamp(0,1)).convert("RGB")

                    true_label = idx_to_name[int(y[i].cpu().item())]
                    pred_label = idx_to_name[int(pred[i].cpu().item())]
                    row_probs = [float(probs[i, j].cpu().item()) for j in range(len(train_ds.classes))]

                    table.add_data(wandb.Image(pil_img), true_label, pred_label, *row_probs)
                    rows_added += 1

    
    test_f1 = f1_score(test_targets, test_preds,average="macro")
    test_acc = correct / total
    print(f"TEST acc = {test_acc:.3f} TEST f1 = {test_f1:.3f}")

    wandb.log({
        "test/acc": test_acc,
        "test/f1": test_f1,
        "examples": table,
    })

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model_path = out / "resnet18_multiclass.pt"
    torch.save(model.state_dict(), out/"resnet18_multiclass.pt")
    print("Saved:", out/"resnet18_multiclass.pt")
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
