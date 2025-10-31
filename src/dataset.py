import os
import shutil

import kagglehub
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

IMAGE_TYPE_COLOR = "/plantvillage dataset/color/"
IMAGE_TYPE_SEGMENTED = "/plantvillage dataset/segmented/"
IMAGE_TYPE_GRAYSCALE = "/plantvillage dataset/grayscale/"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_OUT_DIR = os.path.join(PROJECT_ROOT, "data")

DATA_TRANSFORMS = {
    "train": transforms.Compose(
        [
            # Add transforms
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
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
]


def get_data_path(img_type=IMAGE_TYPE_COLOR):
    return kagglehub.dataset_download("abdallahalidev/plantvillage-dataset") + img_type


def split_data():
    root = get_data_path()

    print("splitting dataset")

    files = []

    labels = []

    for label in CLASSES:
        label_dir = os.path.join(root, label)
        for img in os.listdir(label_dir):
            files.append(os.path.join(label_dir, img))
            labels.append(label)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        files, labels, stratify=labels, test_size=0.2, random_state=42
    )

    copy_files(train_paths, train_labels, "train")
    copy_files(test_paths, test_labels, "test")


def copy_files(paths, labels, split):
    print(f"copying {split}...")
    for path, label in tqdm(zip(paths, labels)):
        dest_dir = os.path.join(SPLIT_OUT_DIR, split, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(path, dest_dir)


def get_train_dataset(transform=None):
    train_path = os.path.join(SPLIT_OUT_DIR, "train")

    if not os.path.isdir(train_path) or len(os.listdir(train_path)) == 0:
        split_data()

    return ImageFolder(root=train_path, transform=transform)


if __name__ == "__main__":
    split_data()
