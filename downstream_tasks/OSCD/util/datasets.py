import os
import pandas as pd
import numpy as np
import warnings
import random
from glob import glob
import json
from typing import Any, Optional, List

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio import logging

from pathlib import Path

import skimage.io as io

log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}
class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class CustomDatasetFromImages(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class FMoWTemporalStacked(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
    
    def __init__(self, csv_path: str, transform: Any):
        """
        Creates Dataset for temporal RGB image classification. Stacks images along temporal dim.
        Usually used for fMoW-RGB-temporal dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion
        """
        super().__init__(in_c=9)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

        self.min_year = 2002

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        splt = single_image_name_1.rsplit('/', 1)
        base_path = splt[0]
        fname = splt[1]
        suffix = fname[-15:]
        prefix = fname[:-15].rsplit('_', 1)
        regexp = '{}/{}_*{}'.format(base_path, prefix[0], suffix)
        temporal_files = glob(regexp)
        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_tensor_1 = self.transforms(img_as_img_1)  # (3, h, w)

        img_as_img_2 = Image.open(single_image_name_2)
        img_as_tensor_2 = self.transforms(img_as_img_2)  # (3, h, w)

        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_3 = self.transforms(img_as_img_3)  # (3, h, w)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        img = torch.cat((img_as_tensor_1, img_as_tensor_2, img_as_tensor_3), dim=0)  # (9, h, w)
        return (img, single_image_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromImagesTemporal(SatelliteDataset):
    def __init__(self, csv_path: str):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=3)

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Scale(224),
            transforms.RandomCrop(224),
        ])
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)

        self.timestamp_arr = np.asarray(self.data_info.iloc[:, 2])
        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))

        self.min_year = 2002  # hard-coded for fMoW

        mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        self.normalization = transforms.Normalize(mean, std)
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Scale(224)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        suffix = single_image_name_1[-15:]
        prefix = single_image_name_1[:-15].rsplit('_', 1)
        regexp = '{}_*{}'.format(prefix[0], suffix)
        regexp = os.path.join(self.dataset_root_path, regexp)
        single_image_name_1 = os.path.join(self.dataset_root_path, single_image_name_1)
        temporal_files = glob(regexp)

        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_img_2 = Image.open(single_image_name_2)
        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_1 = self.totensor(img_as_img_1)
        img_as_tensor_2 = self.totensor(img_as_img_2)
        img_as_tensor_3 = self.totensor(img_as_img_3)
        del img_as_img_1
        del img_as_img_2
        del img_as_img_3
        img_as_tensor_1 = self.scale(img_as_tensor_1)
        img_as_tensor_2 = self.scale(img_as_tensor_2)
        img_as_tensor_3 = self.scale(img_as_tensor_3)
        try:
            if img_as_tensor_1.shape[2] > 224 and \
                    img_as_tensor_2.shape[2] > 224 and \
                    img_as_tensor_3.shape[2] > 224:
                min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w],
                    img_as_tensor_2[..., :min_w],
                    img_as_tensor_3[..., :min_w]
                ], dim=-3)
            elif img_as_tensor_1.shape[1] > 224 and \
                    img_as_tensor_2.shape[1] > 224 and \
                    img_as_tensor_3.shape[1] > 224:
                min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w, :],
                    img_as_tensor_2[..., :min_w, :],
                    img_as_tensor_3[..., :min_w, :]
                ], dim=-3)
            else:
                img_as_img_1 = Image.open(single_image_name_1)
                img_as_tensor_1 = self.totensor(img_as_img_1)
                img_as_tensor_1 = self.scale(img_as_tensor_1)
                img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_1, img_as_tensor_1], dim=-3)
        except:
            print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
            assert False

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        img_as_tensor = self.transforms(img_as_tensor)
        img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
        del img_as_tensor
        img_as_tensor_1 = self.normalization(img_as_tensor_1)
        img_as_tensor_2 = self.normalization(img_as_tensor_2)
        img_as_tensor_3 = self.normalization(img_as_tensor_3)

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        ts3 = self.parse_timestamp(single_image_name_3)

        ts = np.stack([ts1, ts2, ts3], axis=0)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len


#########################################################
# SENTINEL DEFINITIONS
#########################################################


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 years: Optional[List[int]] = [*range(2000, 2021)],
                 categories: Optional[List[str]] = None,
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id', 'timestamp'])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]


        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]

        images = self.open_image('/home/ps/Documents/data/'+selection['image_path'])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['category'])
        # labels = torch.zeros(1,1)
        # print(labels)

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            'images': images,
            'labels': labels,
            'image_ids': selection['image_id'],
            'timestamps': selection['timestamp']
        }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)
def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        # target = target.reshape(1,19)
        return target
class BigEarthNetFinetuneDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    # mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
    #         1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
    #         1972.62420416,  582.72633433,  1732.16362238, 1247.91870117]
    # std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
    #        948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
    #        1364.38688993,  472.37967789,   1310.36996126, 1087.6020813]

    mean = [340.76769064,429.9430203,614.21682446,590.23569706,
                 950.68368468,1792.46290469,2075.46795189,2218.94553375,
                 2266.46036911,2246.0605464,1594.42694882,1009.32729131]
    std = [554.81258967,572.41639287,582.87945694,675.88746967,
        729.89827633,1096.01480586,1273.45393088,1365.45589904,
        1356.13789355,1302.3292881,1079.19066363,818.86747235]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 #years: Optional[List[int]] = [*range(2000, 2021)],
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :csv_path: the txt file loads basic information of dataset
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=12)
        # Filter by category
        # self.categories = CATEGORIES
        # if categories is not None:
        #     self.categories = categories
        #     self.df = self.df.loc[categories]
        # self.indices = self.df.index.unique().to_numpy()

        self.transform = transform
        self.get_multihot_new = get_multihot_new
        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)
        #self.txt_root = '/media/ps/sda1/LXY/SatMAE-main/SatMAE-main/bigearthnet_txt'
        self.image_root =Path( '/home/ps/Documents/data/bigearthnet_cor/')
        #self.image_root_2 = '/home/ps/Documents/data/bigearthnet_cor/'
        #self.label_root = Path('/home/ps/Documents/data/BigEarthNet-v1.0/')

        self.samples = []
        #self.labels_sample=[]
        with open(csv_path) as f:
            for patch_id in f.read().splitlines():
                #self.samples.append(self.image_root + patch_id + '/' + patch_id + '.tif')
                self.samples.append(self.image_root / patch_id)
                #self.labels_sample.append(self.label_root/patch_id)
                #print(self.samples)

    def __len__(self):
        return len(self.samples)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, index):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        path = self.samples[index]  
        patch_id = path.name #S2A_MSIL2A_20180527T093041_25_79
        
        images = self.open_image(path  / f'{patch_id}.tif')

        # images = self.open_image('I:/bigearthnet_cor'+selection['image_path'])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]
        
        with open(  path / f'{patch_id}_labels_metadata.json', 'r') as f:
            labels = json.load(f)['labels']
        targets = self.get_multihot_new(labels)
        # print(targets.shape)
        # targets.transpose(0,1)
        
        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]
        sample = {
            'images': images,
            'labels': targets,
            # 'image_ids': selection['image_id'],
            # 'timestamps': selection['timestamp']
        }
        return img_as_tensor, targets
    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)

    #def get_multihot_new(labels):
     #   target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
      #  for label in labels:
       #     if label in GROUP_LABELS:
        #        target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
         #   elif label not in set(NEW_LABELS):
          #      continue
           # else:
            #    target[NEW_LABELS.index(label)] = 1
       # return target

class BigEarthNetImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433,  1732.16362238, 1247.91870117]
    std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  472.37967789,   1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 #years: Optional[List[int]] = [*range(2000, 2021)],
                 categories: Optional[List[str]] = None,
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=12)
        self.df = pd.read_csv(csv_path) \
            .sort_values(['category'])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        # if years is not None:
        #     self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
        #     self.df = self.df[self.df['year'].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f'BigEarthNet label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):

        # with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            # img = data.read()  # (c, h, w)

        img = io.imread(img_path)
        # return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]

        images = self.open_image('/home/ps/Documents/data/bigearthnet_cor/'+selection['image_path'])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['category'])
        # print(labels)

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            'images': images,
            'labels': labels,
            # 'image_ids': selection['image_id'],
            # 'timestamps': selection['timestamp']
        }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)

class EuroSat(SatelliteDataset):
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

    def __init__(self, file_path, transform, masked_bands=None, dropped_bands=None):
        """
        Creates dataset for multi-spectral single image classification for EuroSAT.
        :param file_path: path to txt file containing paths to image data for EuroSAT.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(13)
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        self.img_paths = [row.split()[0] for row in data]
        self.labels = [int(row.split()[1]) for row in data]

        self.transform = transform

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.img_paths)

    def open_image(self, img_path):
        with rasterio.open('/home/ps/Documents/data/'+img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = self.open_image(img_path)  # (h, w, c)
        if self.masked_bands is not None:
            img[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        img_as_tensor = self.transform(img)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        return img_as_tensor, label


def build_fmow_dataset(is_train: bool, args) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    csv_path = os.path.join(args.train_path if is_train else args.test_path)

    if args.dataset_type == 'rgb':
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transform = CustomDatasetFromImages.build_transform(is_train, args.input_size, mean, std)
        dataset = CustomDatasetFromImages(csv_path, transform)
    elif args.dataset_type == 'temporal':
        dataset = CustomDatasetFromImagesTemporal(csv_path)
    elif args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelIndividualImageDataset(csv_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)
    elif args.dataset_type == 'bigearthnet':
        mean = BigEarthNetImageDataset.mean
        std = BigEarthNetImageDataset.std
        transform =BigEarthNetImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = BigEarthNetImageDataset(csv_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)
    elif args.dataset_type == 'bigearthnet_finetune':
        mean = BigEarthNetFinetuneDataset.mean
        std = BigEarthNetFinetuneDataset.std
        transform =BigEarthNetFinetuneDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = BigEarthNetFinetuneDataset(csv_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)                                                 
    elif args.dataset_type == 'rgb_temporal_stacked':
        mean = FMoWTemporalStacked.mean
        std = FMoWTemporalStacked.std
        transform = FMoWTemporalStacked.build_transform(is_train, args.input_size, mean, std)
        dataset = FMoWTemporalStacked(csv_path, transform)
    elif args.dataset_type == 'euro_sat':
        mean, std = EuroSat.mean, EuroSat.std
        transform = EuroSat.build_transform(is_train, args.input_size, mean, std)
        dataset = EuroSat(csv_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands)
    elif args.dataset_type == 'naip':
        from naip_loader import NAIP_train_dataset, NAIP_test_dataset, NAIP_CLASS_NUM
        dataset = NAIP_train_dataset if is_train else NAIP_test_dataset
        args.nb_classes = NAIP_CLASS_NUM
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset

def build_fmow_dataset_demo(csv_path='/media/ps/sda1/LXY/SatMAE-main/SatMAE-main/bigearthnet_train_demo.txt') -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    # csv_path = os.path.join(args.train_path if is_train else args.test_path)
    mean = BigEarthNetFinetuneDataset.mean
    std = BigEarthNetFinetuneDataset.std
    transform =BigEarthNetFinetuneDataset.build_transform( 128, mean, std)
    dataset = BigEarthNetFinetuneDataset(csv_path, transform)
    print(dataset)

    return dataset


if __name__ == '__main__':
    dataset_train = build_fmow_dataset_demo()
 
   
