# torch
import torch
import torchvision
from torch.utils.data import Dataset
# images
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
# utils
from tqdm import tqdm
import numpy as np
import os

# Dataset Class
class DiabeticRetinopathyDataset(Dataset):
    """
    Each record is ONE eye image.

    Args
    ----
    root_dir   : Path with *.jpeg files
    transform  : torchvision transform to apply
    labels_df  : DataFrame with at least two columns:
                 - 'image' : filename *without* extension
                 - 'level' : integer class label 0-4
    """

    def __init__(self, root_dir, transform=None, labels_df=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform

        # index labels by image stem → grade
        if labels_df is None:
            raise ValueError("labels_df must be provided")
        self.label_map = (
            labels_df.set_index("image")["level"].to_dict()
        )

        # keep only images that have a label
        self.image_paths = [p for p in self.root_dir.glob("*.jpeg")
                            if p.stem in self.label_map]

        if len(self.image_paths) == 0:
            raise RuntimeError("No labelled images found!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path  = self.image_paths[idx]
        img   = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.label_map[path.stem], dtype=torch.long)
        return img, label
    
#------------------------------------------------------------------------------------------
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize = (12, 12))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
#-----------------------------------------------------------------------------------------
