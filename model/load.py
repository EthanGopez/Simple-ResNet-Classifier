import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import Grayscale
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import decode_image

class TumorImageDataset(Dataset):
    """
    Adapted from pytorch docs.
    """
    def __init__(self, annotations_file, img_dir, transform=Grayscale(num_output_channels=1), target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# depending on machine maybe paramaterize num_workers
def load_data(
        annotations_file:str, 
        img_dir:str, 
        batch_size: int = 64, 
        shuffle:bool = False, 
        num_workers: int = 0, 
        pin_memory: bool = False
) -> DataLoader:
    return DataLoader(
        TumorImageDataset(annotations_file=annotations_file, img_dir=img_dir),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
