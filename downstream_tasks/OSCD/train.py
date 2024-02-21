import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr
# Other
import os
import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
from util.pos_embed import interpolate_pos_embed
import numpy as np
import random
import functools
import operator
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
from util.dice_coefficient_loss import dice_loss, build_target
from model.models_vit_tensor_CD import vit_base_patch16
from sync_batchnorm.batchnorm import convert_model

# Global Variables' Definitions
PATH_TO_DATASET = 'data/merge/'
WEIGHT_PATH = './model/checkpoint-14.pth'  #
IS_PROTOTYPE = False

PRETRAIN = False
BATCH_SIZE = 32
PATCH_SIDE = 128
N_EPOCHS = 300
L = 1024
N = 2
TRAIN_STRIDE = int(PATCH_SIDE / 2) - 1  # 50%重叠率裁剪
TYPE = 4  # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands
LOAD_TRAINED = False
DATA_AUG = True
device = "cuda"
DICE = False


# print('DEFINITIONS OK')

class RandomFlip(object):
    """Flip randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        if random.random() > 0.5:
            I1 = I1.numpy()[:, :, ::-1].copy()
            I1 = torch.from_numpy(I1)
            I2 = I2.numpy()[:, :, ::-1].copy()
            I2 = torch.from_numpy(I2)
            label = label.numpy()[:, ::-1].copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        n = random.randint(0, 3)
        if n:
            I1 = sample['I1'].numpy()
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I1 = torch.from_numpy(I1)
            I2 = sample['I2'].numpy()
            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            I2 = torch.from_numpy(I2)
            label = sample['label'].numpy()
            label = np.rot90(label, n, axes=(0, 1)).copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}


# print('UTILS OK')
# Dataset
if DATA_AUG:
    data_transform = tr.Compose([RandomFlip(), RandomRot()])
else:
    data_transform = None

train_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=True, patch_side=PATCH_SIDE, stride=TRAIN_STRIDE,
                                       transform=data_transform)
weights = torch.FloatTensor(train_dataset.weights).to(device)
# weights = torch.FloatTensor([1, 50]).to(device)
print(weights)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, drop_last=True)
test_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=False, patch_side=PATCH_SIDE, stride=TRAIN_STRIDE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32, drop_last=True)
# print('DATASETS OK')
# 0-RGB | 1-RGBIr | 2S-All bands s.t. resulution <= 20m | 3-All bands

if TYPE == 0:
    # net, net_name = Unet(2 * 3, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(3, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'
    # net, net_name = FresUNet(2 * 3, 2), 'FresUNet'
    net, net_name = vit_base_patch16(img_size=128, patch_size=8, in_chans=12), 'stat_FresUNet'
    if PRETRAIN:
        checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % WEIGHT_PATH)
        # checkpoint_model = checkpoint
        checkpoint_model = checkpoint['model']
        state_dict = net.state_dict()
        # for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
        # for k in ['pos_embed']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]

        # load pre-trained model
        net.load_state_dict(checkpoint_model, strict=False)
elif TYPE == 1:
    net, net_name = Unet(2 * 4, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(4, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(4, 2), 'FC-Siam-diff'
    # net, net_name = FresUNet(2 * 4, 2), 'FresUNet'
elif TYPE == 2:
    net, net_name = Unet(2 * 10, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(10, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(10, 2), 'FC-Siam-diff'
    # net, net_name = FresUNet(2 * 10, 2), 'FresUNet'
elif TYPE == 3:
    net, net_name = Unet(2 * 13, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(13, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(13, 2), 'FC-Siam-diff'
    # net, net_name = FresUNet(2 * 13, 2), 'FresUNet'
elif TYPE == 4:

    net, net_name = vit_base_patch16(), 'linear'
    # net, net_name = resnet_vit_base_patch16(), 'resnet'
    if PRETRAIN:
        # checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % WEIGHT_PATH)
        # # checkpoint_model = checkpoint
        # checkpoint_model = checkpoint['model']

        checkpoint = torch.load(
            '/media/ps/sda1/LXY/SatMAE-main/SatMAE-main/SatMAE-main/experiments/pretrain/BE_sep/checkpoint-100.pth',
            map_location='cpu')  # /media/ps/sda1/liyuxuan/change_detection/model/checkpoint-150.pth

        # checkpoint_model = checkpoint
        # checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint_model.items()}

        checkpoint_model = checkpoint['model']

        state_dict = net.state_dict()
        for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            # for k in ['pos_embed_spatial', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight','head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # for k in ['pos_embed']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
        interpolate_pos_embed(net, checkpoint_model)

        # load pre-trained model
        net.load_state_dict(checkpoint_model, strict=False)
        msg = net.load_state_dict(checkpoint_model, strict=False)
        print(msg)
# for n, p in net.named_parameters():
#    if 'block' in n:
#        p.requires_grad = False
# net.cuda()
net = convert_model(net)
net = torch.nn.parallel.DataParallel(net.to(device))

criterion = nn.NLLLoss(weight=weights)  # to be used with logsoftmax output
# criterion = nn.CrossEntropyLoss(weight=weights)

print('NETWORK OK')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Number of trainable parameters:', count_parameters(net))

results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# results_file = "vit_random.txt"



def train(n_epochs=N_EPOCHS, save=True):
    t = np.linspace(1, n_epochs, n_epochs)
    epoch_train_loss = 0 * t
    epoch_train_accuracy = 0 * t
    epoch_train_change_accuracy = 0 * t
    epoch_train_nochange_accuracy = 0 * t
    epoch_train_precision = 0 * t
    epoch_train_recall = 0 * t
    epoch_train_Fmeasure = 0 * t
    epoch_test_loss = 0 * t
    epoch_test_accuracy = 0 * t
    epoch_test_change_accuracy = 0 * t
    epoch_test_nochange_accuracy = 0 * t
    epoch_test_precision = 0 * t
    epoch_test_recall = 0 * t
    epoch_test_Fmeasure = 0 * t
    mean_acc = 0
    best_mean_acc = 0
    fm = 0
    best_fm = 0

    lss = 1000
    best_lss = 1000

    plt.figure(num=1)
    plt.figure(num=2)
    plt.figure(num=3)
    # for n, p in net.named_parameters():
    #     if 'block' in n:
    #         p.requires_grad = False

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-5)

    # optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6)
    #     optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1,verbose=True)

    for epoch_index in tqdm(range(n_epochs)):
        net.train()
        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

        tot_count = 0
        tot_loss = 0
        tot_accurate = 0
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        #         for batch_index, batch in enumerate(tqdm(data_loader)):
        for batch in train_loader:
            I1 = Variable(batch['I1'].float().to(device))
            I2 = Variable(batch['I2'].float().to(device))
            label = torch.squeeze(Variable(batch['label'].to(device)))

            optimizer.zero_grad()
            output = net(I1, I2)
            loss = criterion(output, label.long())
            if DICE is True:
                dice_target = build_target(label.to(torch.int64), 2)
                loss = dice_loss(output, dice_target, multiclass=True)
            loss.backward()
            optimizer.step()

        scheduler.step()

        epoch_train_loss[epoch_index], epoch_train_accuracy[epoch_index], cl_acc, pr_rec = test(train_loader)
        epoch_train_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_train_change_accuracy[epoch_index] = cl_acc[1]
        epoch_train_precision[epoch_index] = pr_rec[0]
        epoch_train_recall[epoch_index] = pr_rec[1]
        epoch_train_Fmeasure[epoch_index] = pr_rec[2]
        print('train_loss:', epoch_train_loss[epoch_index])
        print('train_nochange_accuracy', cl_acc[0])
        print('train_change_accuracy', cl_acc[1])
        print('train_precision', pr_rec[0])
        print('train_recall', pr_rec[1])
        print('train_Fmeasure:', pr_rec[2])

        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            print('\n', file=f)
            print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS), file=f)
            print('train_loss: %s' % epoch_test_loss[epoch_index], file=f)
            print('train_nochange_accuracy: %s' % cl_acc[0], file=f)
            print('train_change_accuracy: %s' % cl_acc[1], file=f)
            print('train_recall: %s' % pr_rec[1], file=f)
            print('train_Fmeasure: %s' % pr_rec[2], file=f)

        #         epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], cl_acc, pr_rec = test(test_dataset)
        epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], cl_acc, pr_rec = test(test_loader)
        epoch_test_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_test_change_accuracy[epoch_index] = cl_acc[1]
        epoch_test_precision[epoch_index] = pr_rec[0]
        epoch_test_recall[epoch_index] = pr_rec[1]
        epoch_test_Fmeasure[epoch_index] = pr_rec[2]
        print('test_loss:', epoch_test_loss[epoch_index])
        print('test_nochange_accuracy', cl_acc[0])
        print('test_change_accuracy', cl_acc[1])
        print('test_precision', pr_rec[0])
        print('test_recall', pr_rec[1])
        print('test_Fmeasure:', pr_rec[2])

        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            print('test_loss: %s' % epoch_test_loss[epoch_index], file=f)
            print('test_nochange_accuracy: %s' % cl_acc[0], file=f)
            print('test_change_accuracy: %s' % cl_acc[1], file=f)
            print('test_recall: %s' % pr_rec[1], file=f)
            print('test_Fmeasure: %s' % pr_rec[2], file=f)
        plt.figure(num=1)
        plt.clf()
        l1_1, = plt.plot(t[:epoch_index + 1], epoch_train_loss[:epoch_index + 1], label='Train loss')
        l1_2, = plt.plot(t[:epoch_index + 1], epoch_test_loss[:epoch_index + 1], label='Test loss')
        plt.legend(handles=[l1_1, l1_2])
        plt.grid()
        #         plt.gcf().gca().set_ylim(bottom = 0)
        plt.gcf().gca().set_xlim(left=0)
        plt.title('Loss')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=2)
        plt.clf()
        l2_1, = plt.plot(t[:epoch_index + 1], epoch_train_accuracy[:epoch_index + 1], label='Train accuracy')
        l2_2, = plt.plot(t[:epoch_index + 1], epoch_test_accuracy[:epoch_index + 1], label='Test accuracy')
        plt.legend(handles=[l2_1, l2_2])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        #         plt.gcf().gca().set_ylim(bottom = 0)
        #         plt.gcf().gca().set_xlim(left = 0)
        plt.title('Accuracy')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=3)
        plt.clf()
        l3_1, = plt.plot(t[:epoch_index + 1], epoch_train_nochange_accuracy[:epoch_index + 1],
                         label='Train accuracy: no change')
        l3_2, = plt.plot(t[:epoch_index + 1], epoch_train_change_accuracy[:epoch_index + 1],
                         label='Train accuracy: change')
        l3_3, = plt.plot(t[:epoch_index + 1], epoch_test_nochange_accuracy[:epoch_index + 1],
                         label='Test accuracy: no change')
        l3_4, = plt.plot(t[:epoch_index + 1], epoch_test_change_accuracy[:epoch_index + 1],
                         label='Test accuracy: change')
        plt.legend(handles=[l3_1, l3_2, l3_3, l3_4])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        #         plt.gcf().gca().set_ylim(bottom = 0)
        #         plt.gcf().gca().set_xlim(left = 0)
        plt.title('Accuracy per class')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=4)
        plt.clf()
        l4_1, = plt.plot(t[:epoch_index + 1], epoch_train_precision[:epoch_index + 1], label='Train precision')
        l4_2, = plt.plot(t[:epoch_index + 1], epoch_train_recall[:epoch_index + 1], label='Train recall')
        l4_3, = plt.plot(t[:epoch_index + 1], epoch_train_Fmeasure[:epoch_index + 1], label='Train Dice/F1')
        l4_4, = plt.plot(t[:epoch_index + 1], epoch_test_precision[:epoch_index + 1], label='Test precision')
        l4_5, = plt.plot(t[:epoch_index + 1], epoch_test_recall[:epoch_index + 1], label='Test recall')
        l4_6, = plt.plot(t[:epoch_index + 1], epoch_test_Fmeasure[:epoch_index + 1], label='Test Dice/F1')
        plt.legend(handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 1)
        #         plt.gcf().gca().set_ylim(bottom = 0)
        #         plt.gcf().gca().set_xlim(left = 0)
        plt.title('Precision, Recall and F-measure')
        display.clear_output(wait=True)
        display.display(plt.gcf())
        every_epoch = 'vit' + str(epoch_index + 1) + '.pth'
        torch.save(net.state_dict(), every_epoch)

        # mean_acc = (epoch_test_nochange_accuracy[epoch_index] + epoch_test_change_accuracy[epoch_index])/2
        # if mean_acc > best_mean_acc:
        #             best_mean_acc = mean_acc
        #             save_str = 'net-best_epoch-' + str(epoch_index + 1) + '_acc-' + str(mean_acc) + '.pth.tar'
        #             torch.save(net.state_dict(), save_str)
        #
        # #         fm = pr_rec[2]
        # fm = epoch_test_Fmeasure[epoch_index]
        # if fm > best_fm:
        #     best_fm = fm
        #     save_str = net_name + str(epoch_index + 1) + '_fm-' + str(fm) + '.pth.tar'
        #     torch.save(net.state_dict(), os.path.join('./pth',save_str))
        #
        # lss = epoch_test_loss[epoch_index]
        # if lss < best_lss:
        #     best_lss = lss
        #     save_str = 'net-best_epoch-' + str(epoch_index + 1) + '_loss-' + str(lss) + '.pth.tar'
        #     torch.save(net.state_dict(), save_str)

        #         print('Epoch loss: ' + str(tot_loss/tot_count))
        if save:
            im_format = 'png'
            #         im_format = 'eps'

            plt.figure(num=1)
            plt.savefig(net_name + '-01-loss.' + im_format)

            plt.figure(num=2)
            plt.savefig(net_name + '-02-accuracy.' + im_format)

            plt.figure(num=3)
            plt.savefig(net_name + '-03-accuracy-per-class.' + im_format)

            plt.figure(num=4)
            plt.savefig(net_name + '-04-prec-rec-fmeas.' + im_format)

            out = {'train_loss': epoch_train_loss[-1],
                   'train_accuracy': epoch_train_accuracy[-1],
                   'train_nochange_accuracy': epoch_train_nochange_accuracy[-1],
                   'train_change_accuracy': epoch_train_change_accuracy[-1],
                   'test_loss': epoch_test_loss[-1],
                   'test_accuracy': epoch_test_accuracy[-1],
                   'test_nochange_accuracy': epoch_test_nochange_accuracy[-1],
                   'test_change_accuracy': epoch_test_change_accuracy[-1]}

    print('pr_c, rec_c, f_meas, pr_nc, rec_nc')
    print(pr_rec)

    return out


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

    # for img_index in dset.names:
    #     I1_full, I2_full, cm_full = dset.get_img(img_index)
    #
    #     s = cm_full.shape
    #
    #     steps0 = np.arange(0, s[0], ceil(s[0] / N))  # [0, PATCH / 2]
    #     steps1 = np.arange(0, s[1], ceil(s[1] / N))
    #     for ii in range(N):
    #         for jj in range(N):
    #             xmin = steps0[ii]
    #             if ii == N - 1:
    #                 xmax = s[0]
    #             else:
    #                 xmax = steps0[ii + 1]
    #             ymin = jj
    #             if jj == N - 1:
    #                 ymax = s[1]
    #             else:
    #                 ymax = steps1[jj + 1]
    #             I1 = I1_full[:, xmin:xmax, ymin:ymax]
    #             I2 = I2_full[:, xmin:xmax, ymin:ymax]
    #             cm = cm_full[xmin:xmax, ymin:ymax]
    #
    #             I1 = Variable(torch.unsqueeze(I1, 0).float()).to(device)
    #             I2 = Variable(torch.unsqueeze(I2, 0).float()).to(device)
    #             cm = Variable(torch.unsqueeze(torch.from_numpy(1.0 * cm), 0).float()).to(device)
    #
    #             output = net(I1, I2)
    for batch in dset:
        I1 = Variable(batch['I1'].float().to(device))
        I2 = Variable(batch['I2'].float().to(device))
        cm = torch.squeeze(Variable(batch['label'].to(device)))
        with torch.no_grad():
            output = net(I1, I2)
        if len(cm.shape) == 2:
            cm = cm.unsqueeze(dim=0)
        # print(output.shape)
        # print(cm.shape)
        loss = criterion(output, cm.long())
        #         print(loss)
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
    net_accuracy = 100 * (tp + tn) / tot_count

    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i], 0.00001)
    print("tp:", tp)
    print("fp:", fp)
    print("tn:", tn)
    print("fn:", fn)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_meas = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)

    pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]

    return net_loss, net_accuracy, class_accuracy, pr_rec


if LOAD_TRAINED:
    net.load_state_dict(torch.load('net_final.pth.tar'))
    print('LOAD OK')
else:
    t_start = time.time()
    out_dic = train()
    t_end = time.time()
    print(out_dic)
    print('Elapsed time:')
    print(t_end - t_start)

if not LOAD_TRAINED:
    # torch.save(net.state_dict(), 'net_final.pth.tar')
    print('SAVE OK')
