# PyTorch
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
from IPython import display
import time
from itertools import chain
import time
import warnings
from pprint import pprint
from dataset import ChangeDetectionDataset
# Models
from model.unet import Unet
from model.siamunet_conc import SiamUnet_conc
from model.siamunet_diff import SiamUnet_diff
from model.fresunet import FresUNet

L = 1024
PATH_TO_DATASET = './OSCD/merge/'
BATCH_SIZE = 32
PATCH_SIDE = 96
N_EPOCHS = 50
TRAIN_STRIDE = int(PATCH_SIDE / 2) - 1  # 50%重叠率裁剪
TYPE = 3  # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

test_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=False, patch_side=PATCH_SIDE, stride=TRAIN_STRIDE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

weights = 10
criterion = nn.NLLLoss(weight=weights)  # to be used with logsoftmax output
if TYPE == 0:
    #     net, net_name = Unet(2*3, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(3, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'
    net, net_name = FresUNet(2 * 3, 2), 'FresUNet'
elif TYPE == 1:
    #     net, net_name = Unet(2*4, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(4, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(4, 2), 'FC-Siam-diff'
    net, net_name = FresUNet(2 * 4, 2), 'FresUNet'
elif TYPE == 2:
    #     net, net_name = Unet(2*10, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(10, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(10, 2), 'FC-Siam-diff'
    net, net_name = FresUNet(2 * 10, 2), 'FresUNet'
elif TYPE == 3:
    #     net, net_name = Unet(2*13, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(13, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(13, 2), 'FC-Siam-diff'
    net, net_name = FresUNet(2 * 13, 2), 'FresUNet'

net.cuda()


# 保存所有结果
def save_test_results(dset):
    for name in tqdm(dset.names):
        with warnings.catch_warnings():
            I1, I2, cm = dset.get_img(name)
            I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
            I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
            out = net(I1, I2)
            _, predicted = torch.max(out.data, 1)
            I = np.stack((255 * cm, 255 * np.squeeze(predicted.cpu().numpy()), 255 * cm), 2)
            io.imsave(f'{net_name}-{name}.png', I)


t_start = time.time()
# save_test_results(train_dataset)
save_test_results(test_dataset)
t_end = time.time()
print('Elapsed time: {}'.format(t_end - t_start))


def kappa(tp, tn, fp, fn):
    N = tp + tn + fp + fn
    p0 = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (N * N)

    return (p0 - pe) / (1 - pe)


def test(dset):
    net.eval()
    tot_loss = 0
    tot_count = 0
    tot_accurate = 0

    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for img_index in tqdm(dset.names):
        I1_full, I2_full, cm_full = dset.get_img(img_index)

        s = cm_full.shape

        for ii in range(ceil(s[0] / L)):
            for jj in range(ceil(s[1] / L)):
                xmin = L * ii
                xmax = min(L * (ii + 1), s[1])
                ymin = L * jj
                ymax = min(L * (jj + 1), s[1])
                I1 = I1_full[:, xmin:xmax, ymin:ymax]
                I2 = I2_full[:, xmin:xmax, ymin:ymax]
                cm = cm_full[xmin:xmax, ymin:ymax]

                I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
                I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0 * cm), 0).float()).cuda()

                output = net(I1, I2)

                loss = criterion(output, cm.long())
                tot_loss += loss.data * np.prod(cm.size())
                tot_count += np.prod(cm.size())

                _, predicted = torch.max(output.data, 1)

                c = (predicted.int() == cm.data.int())
                for i in range(c.size(1)):
                    for j in range(c.size(2)):
                        l = int(cm.data[0, i, j])
                        class_correct[l] += c[0, i, j]
                        class_total[l] += 1

                pr = (predicted.int() > 0).cpu().numpy()
                gt = (cm.data.int() > 0).cpu().numpy()

                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()

    net_loss = tot_loss / tot_count
    net_loss = float(net_loss.cpu().numpy())

    net_accuracy = 100 * (tp + tn) / tot_count

    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i], 0.00001)
        class_accuracy[i] = float(class_accuracy[i].cpu().numpy())

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    dice = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)

    pr_rec = [prec, rec, dice, prec_nc, rec_nc]

    k = kappa(tp, tn, fp, fn)

    return {'net_loss': net_loss,
            'net_accuracy': net_accuracy,
            'class_accuracy': class_accuracy,
            'precision': prec,
            'recall': rec,
            'dice': dice,
            'kappa': k}


results = test(test_dataset)
pprint(results)
