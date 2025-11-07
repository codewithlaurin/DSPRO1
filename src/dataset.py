import argparse
import os
import random
import shutil

from tqdm import tqdm

import kagglehub
from sklearn.model_selection import train_test_split

IMAGE_TYPE_COLOR = "/plantvillage dataset/color/"
IMAGE_TYPE_SEGMENTED = "/plantvillage dataset/segmented/"
IMAGE_TYPE_GRAYSCALE = "/plantvillage dataset/grayscale/"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_OUT_DIR = os.path.join(PROJECT_ROOT, "data")
BINARY_OUT_DIR = os.path.join(PROJECT_ROOT, "data_binary")

CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___healthy",
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Potato___healthy",

]


def get_data_path(img_type=IMAGE_TYPE_COLOR):
    return kagglehub.dataset_download("abdallahalidev/plantvillage-dataset") + img_type

def stratified_train_val_test(files, labels, test_size, val_size, random_state):
    """Return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)."""
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(files, labels, stratify=labels, test_size=test_size, random_state=random_state)

    if val_size == 0:
        return (X_trainval, y_trainval), ([], []), (X_test, y_test)
    
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, stratify=y_trainval, test_size=rel_val, random_state=random_state)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_data(test_size=0.2, val_size=0.1, random_state=27, out_dir=SPLIT_OUT_DIR):
    """Split the multi-class dataset exactly as provided by PlantVillage."""
    root = get_data_path()
    print("splitting dataset (multi-class)")

    files = []
    labels = []

    for label in CLASSES:
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        for img in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img)
            if os.path.isfile(img_path):
                files.append(img_path)
                labels.append(label)

    
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = stratified_train_val_test(files, labels, test_size, val_size, random_state)

    copy_files(train_paths, train_labels, "train", out_dir)
    copy_files(val_paths, val_labels, "val", out_dir)
    copy_files(test_paths, test_labels, "test", out_dir)


def split_balanced_binary(
    test_size=0.2,
    val_size=0.1,
    random_state=27,
    out_dir=BINARY_OUT_DIR,
    balance=True,
):
    """Collapse classes into healthy/unhealthy and optionally balance them."""
    root = get_data_path()
    print("splitting dataset (binary)")

    healthy = []
    unhealthy = []

    for label in CLASSES:
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        bucket = healthy if "healthy" in label.lower() else unhealthy
        for img in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img)
            if os.path.isfile(img_path):
                bucket.append(img_path)

    print(f"found {len(healthy)} healthy and {len(unhealthy)} unhealthy images before balancing")

    rng = random.Random(random_state)
    if balance:
        target = min(len(healthy), len(unhealthy))
        print(f"balancing to {target} samples per class")
        healthy = _sample_without_replacement(healthy, target, rng)
        unhealthy = _sample_without_replacement(unhealthy, target, rng)

    files = healthy + unhealthy
    labels = ["healthy"] * len(healthy) + ["unhealthy"] * len(unhealthy)

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = stratified_train_val_test(files, labels, test_size, val_size, random_state)

    copy_files(train_paths, train_labels, "train", out_dir)
    copy_files(val_paths, val_labels, "val", out_dir)
    copy_files(test_paths, test_labels, "test", out_dir)

    print(
        "final split counts:",
        {
            "train": {
                "healthy": train_labels.count("healthy"),
                "unhealthy": train_labels.count("unhealthy"),
            },
            "test": {
                "healthy": test_labels.count("healthy"),
                "unhealthy": test_labels.count("unhealthy"),
            },
        },
    )

def _sample_without_replacement(items, target, rng):
    """Return a reproducible subset of size `target` (or all items if smaller)."""
    if len(items) <= target:
        return list(items)
    return rng.sample(items, target)


def copy_files(paths, labels, split, out_dir=SPLIT_OUT_DIR):
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    print(f"copying {split} to {split_dir}...")
    for path, label in tqdm(zip(paths, labels), total=len(paths)):
        dest_dir = os.path.join(split_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(path, dest_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Split PlantVillage dataset")
    parser.add_argument(
        "--mode",
        choices=["multiclass", "binary"],
        default="multiclass",
        help="multiclass keeps original labels, binary collapses to healthy/unhealthy",
    )
    parser.add_argument("--out-dir", default=None, help="output directory for the splits")
    parser.add_argument("--test-size", type=float, default=0.2, help="fraction reserved for testing")
    parser.add_argument("--val-size", type=float, default=0.1, help="fraction reserved for validation")
    parser.add_argument("--seed", type=int, default=27, help="random seed")
    parser.add_argument(
        "--balance",
        action="store_true",
        help="balance binary split (ignored in multiclass mode)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir or (BINARY_OUT_DIR if args.mode == "binary" else SPLIT_OUT_DIR)
    if args.mode == "binary":
        split_balanced_binary(
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.seed,
            out_dir=out_dir,
            balance=args.balance,
        )
    else:
        split_data(test_size=args.test_size, val_size=args.val_size, random_state=args.seed, out_dir=out_dir)
