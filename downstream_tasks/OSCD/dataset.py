import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr
# Other
import os
import numpy as np
import random
from skimage import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import floor, ceil, sqrt, exp
import tifffile as tiff
# from osgeo import gdal
import rasterio

NORMALISE_IMGS = True

FP_MODIFIER = 10 # Tuning parameter, use 1 if unsure 权重
TYPE = 4  # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands


# Functions
def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""

    # crop if necesary
    I = I[:s[0], :s[1]]
    si = I.shape

    # pad if necessary
    p0 = max(0, s[0] - si[0])
    p1 = max(0, s[1] - si[1])

    return np.pad(I, ((0, p0), (0, p1)), 'edge')


def read_sentinel_img(path):
    """Read cropped Sentinel-2 image: RGB bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")

    I = np.stack((r, g, b), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_4(path):
    """Read cropped Sentinel-2 image: RGB and NIR bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    I = np.stack((r, g, b, nir), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_leq20(path):
    """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    I = np.stack((r, g, b, nir, ir1, ir2, ir3, nir2, swir2, swir3), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_leq60(path):
    """Read cropped Sentinel-2 image: all bands."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    uv = adjust_shape(zoom(io.imread(path + im_name + "B01.tif"), 6), s)
    wv = adjust_shape(zoom(io.imread(path + im_name + "B09.tif"), 6), s)
    swirc = adjust_shape(zoom(io.imread(path + im_name + "B10.tif"), 6), s)

    I = np.stack((r, g, b, nir, ir1, ir2, ir3, nir2, swir2, swir3, uv, wv, swirc), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_band12(path):
    """Read cropped Sentinel-2 image: all bands."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    uv = adjust_shape(zoom(io.imread(path + im_name + "B01.tif"), 6), s)
    wv = adjust_shape(zoom(io.imread(path + im_name + "B09.tif"), 6), s)

    I = np.stack((uv, b, g, r, ir1, ir2, ir3, nir,nir2, wv, swir2, swir3), axis=2).astype('float')
    # a =  I.mean()
    # if NORMALISE_IMGS:
    #     # I = (I - I.mean()) / I.std()
    #     img = I
    #     kid = (img - img.min(axis=(0, 1), keepdims=True))
    #     mom = (img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True))
    #     img = kid / (mom)
    #     # b = (I - I.mean()) / I.std()


    return I


def read_sentinel_img_trio(path):
    """Read cropped Sentinel-2 image pair and change map."""
    #     read images
    if TYPE == 0:
        I1 = read_sentinel_img(path + '/imgs_1/')
        I2 = read_sentinel_img(path + '/imgs_2/')
    elif TYPE == 1:
        I1 = read_sentinel_img_4(path + '/imgs_1/')
        I2 = read_sentinel_img_4(path + '/imgs_2/')
    elif TYPE == 2:
        I1 = read_sentinel_img_leq20(path + '/imgs_1/')
        I2 = read_sentinel_img_leq20(path + '/imgs_2/')
    elif TYPE == 3:
        I1 = read_sentinel_img_leq60(path + '/imgs_1/')
        I2 = read_sentinel_img_leq60(path + '/imgs_2/')
    elif TYPE == 4:
        I1 = read_sentinel_img_band12(path + '/imgs_1_rect/')
        I2 = read_sentinel_img_band12(path + '/imgs_2_rect/')

    cm = io.imread(path + '/cm/cm.png', as_gray=True) != 0

    # crop if necessary
    s1 = I1.shape
    s2 = I2.shape
    I2 = np.pad(I2, ((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0, 0)), 'edge')

    return I1, I2, cm



def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    #     out = np.swapaxes(I,1,2)
    #     out = np.swapaxes(out,0,1)
    #     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)


class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train=True, patch_side=96, stride=None, use_all_bands=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # basics
        self.transform = transform
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride

        if train:
            # fname = 'train.txt'
            fname = 'train.txt'
        else:
            fname = 'test.txt'

        #         print(path + fname)
        self.names = read_csv(path + fname).columns  # 所有影像的名字
        self.n_imgs = self.names.shape[0]

        n_pix = 0
        true_pix = 0

        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            I1, I2, cm = read_sentinel_img_trio(self.path + im_name)  # 根据设定的TYPE 读取N个波段的影响数据
            self.imgs_1[im_name] = reshape_for_torch(I1)  # [C,H,W] tensor
            self.imgs_2[im_name] = reshape_for_torch(I2)  # [C,H,W] tensor
            self.change_maps[im_name] = cm

            s = cm.shape
            n_pix += np.prod(s)  # 计算乘积 512*512
            true_pix += cm.sum()  # cm label中标签的数量

            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2  # 裁剪后的patch数量

            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i

            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_name,
                                            [self.stride * i, self.stride * i + self.patch_side, self.stride * j,
                                             self.stride * j + self.patch_side],
                                            [self.stride * (i + 1), self.stride * (j + 1)])
                    self.patch_coords.append(current_patch_coords)

        # self.weights = [FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]
        self.weights = [FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]

        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]

        # I1 = (I1 - I1.mean()) / I1.std()
        # I2 = (I2 - I2.mean()) / I2.std()

        I1 = I1.numpy()
        I2 = I2.numpy()  #12,128,128


        kid1 = (I1 - I1.min(axis=(1, 2), keepdims=True))
        mom1 = (I1.max(axis=(1, 2), keepdims=True) - I1.min(axis=(1, 2), keepdims=True))
        I1 = kid1 / (mom1)

        kid2 = (I2 - I2.min(axis=(1, 2), keepdims=True))
        mom2 = (I2.max(axis=(1, 2), keepdims=True) - I2.min(axis=(1, 2), keepdims=True))
        I2 = kid2 / (mom2)

        I1 = torch.tensor(I1)

        I2 = torch.tensor(I2)

        # if NORMALISE_IMGS:
        #     # I = (I - I.mean()) / I.std()
        #     img = I
        #     kid = (img - img.min(axis=(0, 1), keepdims=True))
        #     mom = (img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True))
        #     img = kid / (mom)
        #     # b = (I - I.mean()) / I.std()

        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        # tiff.imwrite(os.path.join("/media/ps/sda1/LXY/data/oscd_visual/label",
        #                           im_name+'_'+str(limits[0])+'_'+str(limits[1])+'_'+str(limits[2])+'_'+str(limits[3])),label)
        label = torch.from_numpy(1 * np.array(label)).float()

        sample = {'I1': I1, 'I2': I2, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

# if __name__ == '__main__':
