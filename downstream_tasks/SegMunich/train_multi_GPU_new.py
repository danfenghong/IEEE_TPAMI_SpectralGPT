import time
import os

import datetime

import torch


from src.models_vit_tensor_CD_2 import vit_base_patch8
from src import UNet
# from src.models_vit_group_channels_seg import vit_base_patch16
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from TUM_128 import SegDataset
import transforms as T
import argparse
import util.misc as misc
import timm
import util.lr_decay as lrd
from util.pos_embed import interpolate_pos_embed
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def create_model(nb_classes, weight_path, pretrain=False):
    model = vit_base_patch8(num_classes=nb_classes)
    # model = vit_large_patch8(num_classes=nb_classes)

    if pretrain:
        checkpoint = torch.load(weight_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % weight_path)
        checkpoint_model = checkpoint['model']
        # checkpoint_model = checkpoint
        state_dict = model.state_dict()
        # for k in ['pos_embed','patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
        for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias','pos_embed_spatial']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        # model.load_state_dict(checkpoint_model, strict=False)
        # print(model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    return model

def create_model_Unet(num_classes, weights, pretrain=False):
    model = UNet(in_channels=12, num_classes=num_classes, base_c=64)
    # model = UPerNet(num_classes=13)
    if pretrain:
        model.load_weights(weights)
    return model

def main(args):
    misc.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    # segmentation nun_classes + background
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    pretrain_path = args.pretrain_path
    warmup_epochs = args.warmup_epochs
    # 用来保存coco_info的文件
    # results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_file = "vit_random.txt"
    # results_file = "unet.txt"

    data_root = args.data_path
    # check data root
    train_dataset = SegDataset(args.data_path, txt_name="train.txt", training=True, data_name="BigEarthNet")
    val_dataset = SegDataset(args.data_path, txt_name="test.txt", training=False, data_name="BigEarthNet")

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        # prefetch_factor=0,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn, drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,

        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + foreground classes
    # model = create_model_Unet(num_classes=num_classes, weights=pretrain_path, pretrain=False)
    model = create_model(nb_classes=num_classes, weight_path=pretrain_path, pretrain=True)
    # model = create_model(nb_classes=13, weight_path='./src/checkpoint-150.pth', pretrain=True)
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]
    # param_groups = model_without_ddp.parameters()
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr, weight_decay=1e-5
    )
    # optimizer = torch.optim.AdamW(
    #     param_groups,
    #     lr=args.lr,weight_decay=0.01)
    # optimizer = torch.optim.SGD(
    #     param_groups,
    #     lr=args.lr,momentum=0.9,weight_decay=args.weight_decay)


    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True,
                                       warmup_epochs=warmup_epochs)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        return

    best_dice = 0.
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.8f}\n"
                f.write(train_info + val_info + "\n\n")

        if args.output_dir:
            # 只在主节点上执行保存权重操作
            save_file = {'model': model_without_ddp.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict(),
                         'args': args,
                         'epoch': epoch}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()

            if args.save_best is True:
                save_on_master(save_file,
                               os.path.join(args.output_dir, 'best_model.pth'))
            else:
                save_on_master(save_file,
                               os.path.join(args.output_dir, 'model_multi_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录(DRIVE)
    parser.add_argument('--data-path', default="/home/ps/Documents/data/TUM_128", help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=12, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument("--pretrain-path", default="./src/checkpoint.pth")
    parser.add_argument("--warmup-epochs", default=15,type=int)
    # 训练的总epoch数
    parser.add_argument('--epochs', default=250, type=int, metavar='N',
                        help='number of total epochs to run')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率，这里默认设置成0.01(使用n块GPU建议乘以n)，如果效果不好可以尝试修改学习率
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_`decay')
    # 只保存dice coefficient值最高的权重
    parser.add_argument('--save-best', default=False, type=bool, help='only save best weights')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./multi_train/', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)


 # python -m torch.distributed.launch --nproc_per_node=4 --master_port=25643 --use_env train_multi_GPU_new.py
