import kagglehub

IMAGE_TYPE_COLOR = "/plantvillage dataset/color/"
IMAGE_TYPE_SEGMENTED = "/plantvillage dataset/segmented/"
IMAGE_TYPE_GRAYSCALE = "/plantvillage dataset/grayscale/"


def get_data_path(img_type=IMAGE_TYPE_COLOR):
    return kagglehub.dataset_download("abdallahalidev/plantvillage-dataset") + img_type


def split_data():
    pass