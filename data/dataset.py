import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from sortedcollections import OrderedSet
import numpy as np
import os
from PIL import Image
from typing import List

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


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
    def __init__(self, root_a: str, root_b: str, size: int = 256):
        super().__init__()
        self.root_a = root_a
        self.root_b = root_b
        self.files = get_image_file_names(root_a) | get_image_file_names(root_b)
        self.size = size


    def __len__(self): 
        return len(self.files)
    

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = TF.resize(img, self.size)
        return img


    @torch.no_grad()
    def __getitem__(self, i):
        name = self.files[i]
        img_a = self.load_image(os.path.join(self.root_a, name))
        img_b = self.load_image(os.path.join(self.root_b, name))

        if random.random() > 0.5:
            img_a, img_b = TF.hflip(img_a), TF.hflip(img_b)

        a, b = TF.to_tensor(img_a), TF.to_tensor(img_b)

        # center on 0
        x = torch.cat([a, b], dim=0) * 2.0 - 1.0

        return x


def tensor_to_image(y: torch.Tensor) -> Image:
    y = y.cpu()
    y = torch.clamp(y, -1, 1)
    y = (y / 2.0 + 0.5) * 255.
    arr = y.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if arr.shape[2] == 3:
        im = Image.fromarray(arr)
    else:
        im = Image.fromarray(arr.squeeze(2), 'L')
    return im

def cat_images(sequence_of_images: List[object]) -> Image:
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    widths, heights = zip(*(i.size for i in sequence_of_images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in sequence_of_images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

def chunk_and_cat(x: torch.Tensor) -> Image:
    return cat_images([tensor_to_image(x[:3]), tensor_to_image(x[3:])])