import os
import numpy as np
import torch.utils.data
from PIL import Image

class DDSM(torch.utils.data.Dataset):
    def __init__(self, root, image_list_path, split, patch_size, transform):
        self.root = root
        with open(image_list_path, 'r') as f:
            self.image_names = list(map(lambda line: line.split()[0], f.readlines()))
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(os.path.join(self.root, image_name))
        min_dim = min(image.size)
        ratio = float(4 * self.patch_size) / min_dim
        new_size = (int(ratio * image.size[0]), int(ratio * image.size[1]))
        image = image.resize(new_size, resample=Image.BILINEAR)
        image = np.asarray(image)
        image = np.broadcast_to(np.expand_dims(image, 2), image.shape + (3,))
        image = self.transform(image)
        return image_name, image
