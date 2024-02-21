from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print('==> Prepping data...')
tile_dir = '/home/supervised_50_100/'  # change it with your path to the tiles
y_fn = '/home/y_50_100.npy'  # change it with your path to the labels
splits_fn = '/home/splits.npy'  # change it with your path to the splits

def clip_and_scale_image(img, img_type='naip', clip_min=0, clip_max=10000):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    if img_type in ['naip', 'rgb']:
        return img / 255
    elif img_type == 'landsat':
        return np.clip(img, clip_min, clip_max) / (clip_max - clip_min)

class RandomFlipAndRotateSinglePatch(object):
    """
    Does what RandomFlipAndRotate except for one anchor.
    """
    def __call__(self, p):
        # Randomly horizontal and vertical flip
        if np.random.rand() < 0.5: p = np.flip(p, axis=2).copy()
        if np.random.rand() < 0.5: p = np.flip(p, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: p = np.rot90(p, k=rotations, axes=(1,2)).copy()
        return p

class ClipAndScaleSinglePatch(object):
    """
    Does what ClipAndScale does except for one anchor.
    """
    def __init__(self, img_type):
        assert img_type in ['naip', 'rgb', 'landsat']
        self.img_type = img_type

    def __call__(self, p):
        p = clip_and_scale_image(p, self.img_type)
        return p


class ToFloatTensorSinglePatch(object):
    """
    Does what ToFloatTensor does except for one anchor.
    """
    def __call__(self, p):
        p = torch.from_numpy(p).float()
        return p


mean = [0.47996818, 0.47633689, 0.42402868]
std = [0.17774512, 0.13703743, 0.11909943]


transform_tr = transforms.Compose([
    ClipAndScaleSinglePatch('naip'),
    RandomFlipAndRotateSinglePatch(),
    
    ToFloatTensorSinglePatch(),
    transforms.Normalize(mean, std),
    transforms.Scale(224),
])
transform_val = transforms.Compose([
    ClipAndScaleSinglePatch('naip'),
    
    ToFloatTensorSinglePatch(),
    transforms.Normalize(mean, std),
    transforms.Scale(224),
])
transform_te = transforms.Compose([
    ClipAndScaleSinglePatch('naip'),
    
    ToFloatTensorSinglePatch(),
    transforms.Normalize(mean, std),
    transforms.Scale(224),
])


# Encode labels
y = np.load(y_fn)
le = LabelEncoder()
le.fit(y)
n_classes = len(le.classes_)
labels = le.transform(y)
NAIP_LABELS = labels
NAIP_CLASS_NUM = n_classes

# Getting train/val/test idxs
splits = np.load(splits_fn)
idxs_tr = np.where(splits == 0)[0]
idxs_val = np.where(splits == 1)[0]
idxs_te = np.where(splits == 2)[0]
idxs_te = np.concatenate((idxs_val, idxs_te))


class NAIP(Dataset):
    def __init__(self, tile_dir, tile_idxs, labels=None,
        transform=None):
        self.in_c = 3
        self.tile_dir = tile_dir
        self.tile_idxs = tile_idxs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.tile_idxs)

    def __getitem__(self, idx):
        p_idx = self.tile_idxs[idx]
        p = np.load(os.path.join(self.tile_dir, '{}tile.npy'.format(p_idx)))
        p = p[:, :, :3]
        p = np.moveaxis(p, -1, 0)
        y = self.labels[p_idx]
        if self.transform:
            p = self.transform(p)
        return (p, y)

# Setting up Datasets
NAIP_train_dataset = NAIP(tile_dir, idxs_tr, labels=labels, transform=transform_tr)
NAIP_test_dataset = NAIP(tile_dir, idxs_te, labels=labels, transform=transform_te)
    
    