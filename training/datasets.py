import os
import torch.utils.data
from PIL import Image


class DDSM(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        self.root = root
        def process_line(line):
            image_name, label = line.strip().split(' ')
            label = int(label)
            return image_name, label
        with open(os.path.join(root, '{}.txt'.format(split)), 'r') as f:
            self.image_list = list(map(process_line, f.readlines()))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name, label = self.image_list[idx]
        image = Image.open(os.path.join(self.root, image_name))
        image = self.transform(image)
        return image, label
