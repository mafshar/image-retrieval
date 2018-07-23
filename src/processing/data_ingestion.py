
import os
import sys
import glob
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch

from os.path import join
from os import listdir
from scipy.misc import imread, imresize, imsave
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms


DATA_DIR = './data'
IMG_DIR = join(DATA_DIR, 'jpg')
IMG_SIZE = 256


def __get_transformations():
    return transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

def load_data(dir_path=IMG_DIR, batch_size=32, num_workers=0):
    try:
        data = HolidayDataset(
            dir_path=dir_path,
            transform=__get_transformations())
        dataloaders = train_val_split_loader(
            dataset=data,
            batch_size=batch_size,
            num_workers=num_workers)
    except OSError:
        print 'Directory not found. Loading data failed. Existing...'
        exit(-1)
    return dataloaders

def train_val_split_loader(dataset, batch_size, num_workers=0):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = 4

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    trainloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers)
    validationloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=validation_sampler,
        num_workers=num_workers)
    return trainloader, validationloader


def __create_data_loader(dataset, batch_size, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

class HolidayDataset(Dataset):

    def __init__(self, dir_path=DATA_DIR, transform=None):
        self.files = [join(dir_path, f) for f in listdir(dir_path) if f.endswith('jpg')]
        self.transform = transform
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img = Image.open(filename)
        if self.transform:
            img = self.transform(img)
        return img
