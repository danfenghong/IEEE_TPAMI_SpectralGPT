import os
import time
import json
import torch
from torchvision import transforms
from src.models_vit_tensor_CD_2 import vit_base_patch16
import numpy as np
from PIL import Image
import skimage.io as io

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def open_image(img_path):
    img = io.imread(img_path)
    return img.astype(np.float32)


def main():
    classes = 1
    weights_path = "checkpoint.pth"
    palette_path = "palette.json"
    # use_CRF = True
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = vit_base_patch16(num_classes=13)

    # delete weights about aux_classifier
    # weights_dict = torch.load(weights_path, map_location='cpu')['model']
    # load weights
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    model.to(device)

    # load image
    image_folder_path = "/TUM/img"

    # 获取图片文件夹中的所有图片文件名
    image_file_names = os.listdir(image_folder_path)

    # 遍历图片文件名
    for image_file_name in image_file_names:
        img = open_image(os.path.join(image_folder_path, image_file_name))
        b = np.mean(img, axis=2)
        b = np.expand_dims(b, axis=2)

        img = np.concatenate((img, b, b), axis=2)

        kid = (img - img.min(axis=(0, 1), keepdims=True))
        mom = (img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True))
        img = kid / (mom)

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = data_transform(img)

        # # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        img = img.cuda()

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            # img_height, img_width = img.shape[-2:]
            # init_img = torch.zeros((1, 3, img_height, img_width), device=device)

            model(img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            for i in range(len(prediction)):
                for j in range(len(prediction[0])):
                    if prediction[i][j] == 1:
                        prediction[i][j] = 21
                    elif prediction[i][j] == 2:
                        prediction[i][j] = 22
                    elif prediction[i][j] == 3:
                        prediction[i][j] = 23
                    elif prediction[i][j] == 4:
                        prediction[i][j] = 31
                    elif prediction[i][j] == 6:
                        prediction[i][j] = 32
                    elif prediction[i][j] == 7:
                        prediction[i][j] = 33
                    elif prediction[i][j] == 8:
                        prediction[i][j] = 41
                    elif prediction[i][j] == 9:
                        prediction[i][j] = 13
                    elif prediction[i][j] == 10:
                        prediction[i][j] = 14
            mask = Image.fromarray(prediction)
            mask.putpalette(pallette)
            mask.save(os.path.join("/TUM/predict/", image_file_name))


if __name__ == '__main__':
    main()
