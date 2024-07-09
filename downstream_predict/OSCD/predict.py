import os
import time
import json

import torch
from torchvision import transforms
from model.models_vit_tensor_CD import vit_base_patch16
import skimage.io as io

import numpy as np
from PIL import Image
# from src import UNet
#
# from src import UNet
# import pydensecrf.densecrf as dcrf


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def open_image(img_path):
    # with rasterio.open(img_path) as data:
    #     img = data.read()  # (c, h, w)
    img = io.imread(img_path)

    # return img.transpose(1, 2, 0).astype(np.float32)
    return img.astype(np.float32)

def main():
    palette_path = "palette.json"

    # assert os.path.exists(weights_path), f"weights {weights_path} not found."
    # assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = vit_base_patch16()
    # model = UNet(in_channels=12, num_classes=13, base_c=64)
    # model = UPerNet(num_classes=13)

    # delete weights about aux_classifier
    # weights_dict = torch.load(weights_path, map_location='cpu')['model']
    # load weights
    checkpoint = torch.load('checkpoint.pth',
                            map_location=device)

    checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    # model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # load image
    image_folder_path1 = "/OSCD/I1"
    image_folder_path2 = "/OSCD/I2"#

    # 获取图片文件夹中的所有图片文件名
    image_file_names = os.listdir(image_folder_path1)

    # 遍历图片文件名
    for image_file_name in image_file_names:
        img1 = open_image(os.path.join(image_folder_path1, image_file_name))
        img2 = open_image(os.path.join(image_folder_path2, image_file_name))

        kid1 = (img1 - img1.min(axis=(0, 1), keepdims=True))
        mom1 = (img1.max(axis=(0, 1), keepdims=True) - img1.min(axis=(0, 1), keepdims=True))
        img1 = kid1 / (mom1)

        kid2 = (img2 - img2.min(axis=(0, 1), keepdims=True))
        mom2 = (img2.max(axis=(0, 1), keepdims=True) - img2.min(axis=(0, 1), keepdims=True))
        img2 = kid2 / (mom2)


            # from pil image to tensor and normalize
        data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        img1 = data_transform(img1)
        img1 = torch.unsqueeze(img1, dim=0)
        img2 = data_transform(img2)
        img2 = torch.unsqueeze(img2, dim=0)

        img1 = img1.cuda()
        img2 = img2.cuda()

        model.eval()  # 进入验证模式
        with torch.no_grad():
                t_start = time_synchronized()
                output = model(img1.to(device),img2.to(device))
                t_end = time_synchronized()
                print("inference time: {}".format(t_end - t_start))

                prediction = output.argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                # print(prediction)

                mask = Image.fromarray(prediction)
                mask.putpalette(pallette)
                mask.save(os.path.join("/OSCD/predict_1", image_file_name))



if __name__ == '__main__':
    main()
