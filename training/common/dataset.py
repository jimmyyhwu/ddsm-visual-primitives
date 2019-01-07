import os
import torch.utils.data
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms


IMAGE_SIZE_TO_ANALYZE = 512


def get_default_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform


def preprocess_image(path, target_size, transform):
    image = Image.open(path)
    min_dim = min(image.size)
    ratio = target_size / min_dim
    new_size = (int(ratio * image.size[0]), int(ratio * image.size[1]))
    image = image.resize(new_size, resample=Image.BILINEAR)  # image shape is now (~1500, 896)
    delta_w = target_size - new_size[0]
    delta_h = target_size - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    image = ImageOps.expand(image, padding)
    image = np.asarray(image)
    image = np.broadcast_to(np.expand_dims(image, 2), image.shape + (3,))  # image shape is now (~1500, 896, 3)
    image = transform(image)  # image shape is now (3, ~1500, 896) and a it is a tensor
    return image


def preprocess_image_default(path):
    transform = get_default_transform()
    return preprocess_image(path, IMAGE_SIZE_TO_ANALYZE, transform)


class DDSM(torch.utils.data.Dataset):
    def __init__(self, root, image_list_path, target_size, transform):
        self.root = root
        name2class = {
            'normal': 0,
            'benign': 1,
            'cancer': 2,
        }
        with open(image_list_path, 'r') as f:
            self.images = [(line.strip(), name2class[line.strip()[:6]]) for line in f.readlines()]
        self.image_names = [filename for filename, ground_truth in self.images]
        self.target_size = target_size
        self.transform = transform

        classes, class_count = np.unique([label for _, label in self.images], return_counts=True)
        if (classes != [0, 1, 2]).all():
            raise RuntimeError("DDSM Dataset: classes are missing or in wrong order")
        self.weight = 1 / (class_count / np.amin(class_count))

        print("Dataset balance (normal, benign, malignant):", class_count)

    @staticmethod
    def create_full_image_dataset(split):
        raw_image_dir = '../data/ddsm_raw'
        image_list = '../data/ddsm_raw_image_lists/' + split + '.txt'
        dataset = DDSM(raw_image_dir, image_list, IMAGE_SIZE_TO_ANALYZE, get_default_transform())
        return dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name, ground_truth = self.images[idx]
        image = preprocess_image(os.path.join(self.root, image_name), self.target_size, self.transform)
        return image, ground_truth
