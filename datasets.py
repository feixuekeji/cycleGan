import glob
import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def to_rgb(image):
    # rgb_image = Image.new("RGB", image.size)
    # rgb_image.paste(image)
    image = image.convert('RGB')
    return image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        # image_c is image_A HR image

        image_B = image_C = Image.open(self.files_B[index % len(self.files_B)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        if image_C.mode != "RGB":
            image_C = to_rgb(image_C)
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_C = self.transform(image_C)

        return {"A": item_A, "B": item_B, "C": item_C}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
