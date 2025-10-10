import os
import shutil
from tqdm import tqdm

import kagglehub
from sklearn.model_selection import train_test_split

IMAGE_TYPE_COLOR = "/plantvillage dataset/color/"
IMAGE_TYPE_SEGMENTED = "/plantvillage dataset/segmented/"
IMAGE_TYPE_GRAYSCALE = "/plantvillage dataset/grayscale/"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_OUT_DIR = os.path.join(PROJECT_ROOT, "data")

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

    print('splitting dataset')

    files = []

    labels = []

    for label in CLASSES:
        label_dir = os.path.join(root, label)
        for img in os.listdir(label_dir):
            files.append(os.path.join(label_dir, img))
            labels.append(label)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        files, labels, stratify=labels, test_size=0.3, random_state=27
    )

    test_paths, val_paths, test_labels, val_labels = train_test_split(
        temp_paths, temp_labels, stratify=temp_labels, test_size=1 / 3, random_state=27
    )

    copy_files(train_paths, train_labels, 'train')
    copy_files(test_paths, test_labels, 'test')
    copy_files(val_paths, val_labels, 'val')


def copy_files(paths, labels, split):
    print(f'copying {split}...')
    for path, label in tqdm(zip(paths, labels)):
        dest_dir = os.path.join(SPLIT_OUT_DIR, split, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(path, dest_dir)

if __name__ == "__main__":
    split_data()
