import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
# from osgeo import gdal
import rasterio
import skimage.io as io
from imgaug import augmenters as iaa
import torchvision.transforms.functional as transF

mean_std_dict = {
    'BigEarthNet': ['BigEarthNet',
                     #[708.60993377, 780.18614848, 920.97794044, 908.09042059, 1259.50045652,
                     #             2043.74497623, 2293.325926, 2445.49642642, 2460.64317951, 2446.06940002,
                      #            1491.30623544, 963.26966582],
                    #[304.79360028, 411.16626381, 434.15785986, 500.10627497, 493.87129362,
                    # 636.04945099, 699.74150505, 808.26186653, 718.71649735, 591.19044836,
                  #  431.50743065, 363.09343017],
 [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
          1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433,  1732.16362238, 1247.91870117],
 [633.15169573, 650.2842772, 712.12507725, 965.23119807,
            948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 1310.36996126, 1087.602081],
                    '.tif']
}


class SegDataset(data.Dataset):
    def __init__(self, image_root, txt_name: str = "train.txt", training=False, data_name="BigEarthNet"):
        super(SegDataset, self).__init__()
        assert os.path.exists(image_root), "path '{}' does not exist.".format(image_root)
        image_dir = os.path.join(image_root, 'bigearthnet_cor')
        mask_dir = os.path.join(image_root, 'clc_maps/clc_maps_compressed')

        txt_path = os.path.join(image_root, "ImageSets", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.training = training
        self.images = [os.path.join(image_dir, x, x + ".tif") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".tif") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.name, self.mean, self.std, self.shuffix = mean_std_dict[data_name]
        # 影像预处理方法
        self.transform = iaa.Sequential([

            # iaa.Affine(scale=(0.5, 2.0)),
            # iaa.Crop(px=(48,120)),
            # iaa.Resize(120),
            iaa.Rot90([0, 1, 2, 3]),
            iaa.VerticalFlip(p=0.5),
            iaa.HorizontalFlip(p=0.5),
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        img = open_image(self.images[index])
        # width = imgs.RasterXSize
        # height = imgs.RasterYSize
        # img = imgs.ReadAsArray(0, 0, width, height).transpose(1, 2, 0)

        target = np.array(Image.open(self.masks[index]).convert("P"))
        # width = labels.RasterXSize
        # height = labels.RasterYSize
        # target = labels.ReadAsArray(0, 0, width, height)
        target[target > 33] -= 1

        if self.training:
            # 利用_load_maps获取得到的distance_map和angle_map
            img, target = self.transform(image=img, segmentation_maps=np.stack(
                (target[np.newaxis, :, :], target[np.newaxis, :, :]), axis=-1))
            target = target[0, :, :, 0]
        img, target = torch.tensor(img.copy()).permute(2, 0, 1), torch.tensor(target.copy()).long()
        # 标准化
        img = transF.normalize(img, self.mean, self.std)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def open_image(img_path):
    # with rasterio.open(img_path) as data:
    #     img = data.read()  # (c, h, w)
    img = io.imread(img_path)

    # return img.transpose(1, 2, 0).astype(np.float32)
    return img.astype(np.float32)
# dataset = Massachusetts(building_root="F:/Massachusetts_buliding_data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
