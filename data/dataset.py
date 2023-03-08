import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from sortedcollections import OrderedSet
import numpy as np
import os
from PIL import Image
from typing import Tuple


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def resize_reformat_files(root_in, root_out, size):
    image_names = get_image_file_names(root_in)
    os.makedirs(root_out, exist_ok=True)
    for name in image_names:
        path_in = os.path.join(root_in, name)
        img = Image.open(path_in).convert('RGB')
        img = TF.resize(img, size)
        name_out = name.replace('.png', '.jpeg')
        path_out = os.path.join(root_out, name_out)
        img.save(path_out)

def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_file_names(dir) -> set[str]:
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                # path = os.path.join(root, fname)
                images.append(fname)
    return OrderedSet(images)


class PairedDataset(Dataset):
    def __init__(self, roots: Tuple[str, str], size: int = 256, channels=(3, 3)):
        super().__init__()
        self.root_a, self.root_b = roots
        self.channels_a, self.channels_b = channels
        self.files = get_image_file_names(self.root_a) | get_image_file_names(self.root_b)
        self.size = size

    def __len__(self): 
        return len(self.files)

    def load_image(self, path) -> Image:
        img = Image.open(path).convert('RGB')
        img = TF.resize(img, self.size)
        return img

    @torch.no_grad()
    def __getitem__(self, i) -> torch.Tensor:
        name = self.files[i]
        img_a = self.load_image(os.path.join(self.root_a, name))
        img_b = self.load_image(os.path.join(self.root_b, name))

        if random.random() > 0.5:
            img_a, img_b = TF.hflip(img_a), TF.hflip(img_b)

        a, b = TF.to_tensor(img_a), TF.to_tensor(img_b)

        # center on 0
        x = torch.cat([a[:self.channels_a], b[:self.channels_b]], dim=0) * 2.0 - 1.0

        return x



