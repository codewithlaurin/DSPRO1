import argparse
import os
import random
import shutil

import kagglehub
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

IMAGE_TYPE_COLOR = "/plantvillage dataset/color/"
IMAGE_TYPE_SEGMENTED = "/plantvillage dataset/segmented/"
IMAGE_TYPE_GRAYSCALE = "/plantvillage dataset/grayscale/"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_OUT_DIR = os.path.join(PROJECT_ROOT, "data")
BINARY_OUT_DIR = os.path.join(PROJECT_ROOT, "data_binary")


def random_noise(img: Image.Image, threshold: int = 5):
    arr = np.array(img)

    mask = (
        (arr[:, :, 0] < threshold)
        & (arr[:, :, 1] < threshold)
        & (arr[:, :, 2] < threshold)
    )

    if not np.any(mask):
        return img

    dilated_mask = binary_dilation(mask, iterations=3)

    mean_color = [[50, 100, 50], [80, 70, 60], [185, 180, 170]]
    idx_color = np.random.randint(0, 3)
    epsilon = 25

    noise = np.random.normal(mean_color[idx_color], epsilon, arr.shape).astype(np.uint8)

    arr[dilated_mask] = noise[dilated_mask]

    return Image.fromarray(arr)


DATA_TRANSFORMS = {
    "train": transforms.Compose(
        [
            transforms.Lambda(lambda x: random_noise(x, 5)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

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
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
]


def get_data_path(img_type=IMAGE_TYPE_SEGMENTED):
    return kagglehub.dataset_download("abdallahalidev/plantvillage-dataset") + img_type


def split_data(test_size=0.2, random_state=42, out_dir=SPLIT_OUT_DIR):
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

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        files, labels, stratify=labels, test_size=test_size, random_state=random_state
    )

    copy_files(train_paths, train_labels, "train", out_dir)
    copy_files(test_paths, test_labels, "test", out_dir)


def split_hold_out(test_size=0.2, random_state=42, out_dir=SPLIT_OUT_DIR):
    """Split the multi-class dataset exactly as provided by PlantVillage. With a hold out val set"""

    root = get_data_path()
    print("splitting dataset (multi-class + val)")

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

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        files, labels, stratify=labels, test_size=test_size, random_state=random_state
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        stratify=train_labels,
        test_size=test_size,
        random_state=random_state,
    )

    copy_files(train_paths, train_labels, "train", out_dir)
    copy_files(test_paths, test_labels, "test", out_dir)
    copy_files(val_paths, val_labels, "val", out_dir)


def split_balanced_binary(
    test_size=0.2,
    random_state=42,
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

    print(
        f"found {len(healthy)} healthy and {len(unhealthy)} unhealthy images before balancing"
    )

    rng = random.Random(random_state)
    if balance:
        target = min(len(healthy), len(unhealthy))
        print(f"balancing to {target} samples per class")
        healthy = _sample_without_replacement(healthy, target, rng)
        unhealthy = _sample_without_replacement(unhealthy, target, rng)

    files = healthy + unhealthy
    labels = ["healthy"] * len(healthy) + ["unhealthy"] * len(unhealthy)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        files,
        labels,
        stratify=labels,
        test_size=test_size,
        random_state=random_state,
    )

    copy_files(train_paths, train_labels, "train", out_dir)
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


def get_train_dataset(transform=None, binary=False):
    train_path = os.path.join(BINARY_OUT_DIR if binary else SPLIT_OUT_DIR, "train")

    if not os.path.isdir(train_path) or len(os.listdir(train_path)) == 0:
        if binary:
            split_balanced_binary()
        else:
            split_data()

    return ImageFolder(root=train_path, transform=transform)


def get_test_dataset(binary=False):
    test_path = os.path.join(BINARY_OUT_DIR if binary else SPLIT_OUT_DIR, "test")

    if not os.path.isdir(test_path) or len(os.listdir(test_path)) == 0:
        if binary:
            split_balanced_binary()
        else:
            split_data()

    return ImageFolder(root=test_path, transform=DATA_TRANSFORMS["val"])


def get_val_dataset(binary=False):
    val_path = os.path.join(BINARY_OUT_DIR if binary else SPLIT_OUT_DIR, "val")

    if not os.path.isdir(val_path) or len(os.listdir(val_path)) == 0:
        return None

    return ImageFolder(root=val_path, transform=DATA_TRANSFORMS["val"])


def parse_args():
    parser = argparse.ArgumentParser(description="Split PlantVillage dataset")
    parser.add_argument(
        "--mode",
        choices=["multiclass", "binary", "hold-out"],
        default="multiclass",
        help="multiclass keeps original labels, binary collapses to healthy/unhealthy",
    )
    parser.add_argument(
        "--out-dir", default=None, help="output directory for the splits"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="fraction reserved for testing"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--balance",
        action="store_true",
        help="balance binary split (ignored in multiclass mode)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir or (
        BINARY_OUT_DIR if args.mode == "binary" else SPLIT_OUT_DIR
    )
    if args.mode == "binary":
        split_balanced_binary(
            test_size=args.test_size,
            random_state=args.seed,
            out_dir=out_dir,
            balance=args.balance,
        )
    elif args.mode == "hold-out":
        split_hold_out(
            test_size=args.test_size, random_state=args.seed, out_dir=out_dir
        )
    else:
        split_data(test_size=args.test_size, random_state=args.seed, out_dir=out_dir)
