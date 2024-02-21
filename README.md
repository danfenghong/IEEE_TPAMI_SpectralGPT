# SpectralGPT (TPAMI) 
**[Paper](https://arxiv.org/abs/2311.07113)** 

This is the official repository for the paper 
"_SpectralGPT: Spectral Remote Sensing Foundation Model_".  

## Preparation
Install Python dependencies by running:
```shell
pip install -r requirements.txt
```
## Training SpectralGPT
The pretraining experiments were ran on 8 NVIDIA GeForce RTX 4090 GPUs.
### Pretrain Dataset: fMoW-Sentinel & BigEarthNet 
You can download the official fMoW-Sentinel dataset [here](https://purl.stanford.edu/vg497cb6002). 
Try this [link](https://searchworks.stanford.edu/view/vg497cb6002) if the previous one doesn't display correctly.

You can download the official BigEarthNet dataset [here](https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz). 

### Finetune Dataset: EuroSAT & OSCD & SegMunich
You can download the official EuroSAT dataset [here](https://github.com/phelber/EuroSAT#eurosat-land-use-and-land-cover-classification-with-sentinel-2) for finetuning the pretrained model on classification tasks.  

You can download the official OSCD dataset [here](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) for finetuning the pretrained model on change detection tasks. 

You can download the official SegMunich dataset we collected [here](https://pan.baidu.com/s/1ouz_FVOdENjkAZRajjkjVw?pwd=994z) for finetuning the pretrained model on semantic segmentation tasks.

Dataset                  |Use| Link |  |
---------------------- | -------------- | -------- | -------- 
fMoW-Sentinel  |     pretrain     | [download](https://purl.stanford.edu/vg497cb6002) 
BigEarthNet |    pretrain & finetune    | [download](https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz) |    
EuroSAT  |     finetune     | [download](https://github.com/phelber/EuroSAT#eurosat-land-use-and-land-cover-classification-with-sentinel-2) 
OSCD  |       finetune   | [download](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) 
SegMunich  |         finetune | [download](https://pan.baidu.com/s/1ouz_FVOdENjkAZRajjkjVw?pwd=994z)  

### Pretraining
For pretraining on fMoW-Sentinel Dataset, this is the default command:
```shell
torchrun --nproc_per_node=8 main_pretrain.py \
--master_port=29501 \
--wandb spectralgpt_pretrain_stage1 \
--batch_size 16 --accum_iter 32 --blr 0.0002 \
--epochs 200 --warmup_epochs 20 --num_workers 16 \
--input_size 96 --patch_size 8 \
--mask_ratio 0.90 \
--model_type tensor \
--model mae_vit_base_patch8_96 \
--dataset_type sentinel --dropped_bands 10 \
--train_path .txt_file/train_result_demo.csv \
--output_dir .experiments/pretrain_fmow \
--log_dir .experiments/pretrain_fmow
```

For continual pretraining on BigEarthNet Dataset, this is the default command:
```shell
torchrun --nproc_per_node=8 main_pretrain.py \
--master_port=29502 \
--wandb spectralgpt_pretrain_stage2 \
--batch_size 16 --accum_iter 32 --blr 0.0001 \
--epochs 200 --warmup_epochs 20 --num_workers 16 \
--input_size 128 --patch_size 8 \
--mask_ratio 0.90 \
--model_type tensor \
--dataset_type bigearthnet \
--model mae_vit_base_patch8_128 \
--train_path .txt_file/bigearthnet_pretrain_result_demo.csv \
--resume_different_size .experiments/pretrain_fmow/checkpoint-199.pth \
--output_dir .experiments/pretrain_BEN \
--log_dir .experiments/pretrain_BEN
```

To resume a pretraining job, you can use `--resume PATH/TO/CKPT.PTH` 
(eg: `--resume .experiments/pretrain/checkpoint-10.pth`).


### Finetuning
To finetune on EuroSAT, the basic command is:
```shell
torchrun --nproc_per_node=2 main_finetune.py \
--wandb eurosat_finetune \
--batch_size 16 --accum_iter 8 --blr 0.0002 \
--epochs 150 --num_workers 16 \
--input_size 128 --patch_size 8  \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
--model_type tensor \
--model mae_vit_base_patch8_128 \
--dataset_type euro_sat --dropped_bands 10 \
--train_path .txt_file/train_euro_result.txt \
--test_path .txt_file/val_euro_result.txt \
--output_dir /home/experiments/finetune/eurosat \
--log_dir ./experiments/finetune/eurosat \
--finetune ./experiments/pretain/SpectralGPT+.pth
```


To finetune on BigEarthNet, please replace `engine_finetune`(line 44-45) with `engine_finetune_BE`(line 46-47) in the [main_finetune.py](./main_finetune.py), the basic command is:
```shell
torchrun --nproc_per_node=2 main_finetune.py \
--wandb bigearthnet_finetune \
--batch_size 16 --accum_iter 8 --blr 0.0002 \
--epochs 150 --num_workers 16 \
--input_size 128 --patch_size 8  \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
--model_type tensor \
--model mae_vit_base_patch8_128 \
--dataset_type euro_sat --dropped_bands 10 \
--train_path .txt_file/bigearthnet_train.txt \
--test_path .txt_file/bigearthnet_val.txt \
--output_dir /home/experiments/finetune/BEN \
--log_dir ./experiments/finetune/BEN \
--finetune ./experiments/pretain/SpectralGPT+.pth
```

We also released the codes of change detection on OSCD and semantic segmentation on SegMunich in the [downstream_tasks](./downstream_tasks/) folder.  These codes are easy to use when paired with the correct data and checkpoint paths.

To finetune on OSCD dataset, the basic command is:
```shell
python train.py
```
To finetune on SegMunich dataset, the basic command is:
```shell
python -m torch.distributed.launch --nproc_per_node=2 \
--master_port=25643 --use_env train_multi_GPU_new.py
```
### Model Weights
We have already uploaded our model checkpoints [here](https://zenodo.org/records/8412455).
The [SpectralGPT.pth](https://zenodo.org/records/8412455/files/SpectralGPT.pth?download=1) checkpoint has been trained for 200 epochs on fMoW-Sentinel Dataset and the [SpectralGPT+.pth](https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1) has been continal pretrained on BigEarthNet Dataset for 100 epochs. 


Model                  | | Checkpoint |  |
---------------------- | -------------- | -------- | -------- |
SpectralGPT (200 epochs)  |          | [download](https://zenodo.org/records/8412455/files/SpectralGPT.pth?download=1)  |  |
SpectralGPT+ (100 epochs) |        | [download](https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1) |    |

## Acknowledgements
Pretrain and downstream classification codes from this repository are inspired from the Masked Autoencoders (MAE) [repository](https://github.com/facebookresearch/mae) and SatMAE [repository](https://github.com/sustainlab-group/SatMAE). The downstream pixel-level codes from this repository are inspired from Seasonal Contrast (SeCo) [repository](https://github.com/ServiceNow/seasonal-contrast) and Fully Convolutional Siamese Networks for Change Detection [repository](https://github.com/rcdaudt/fully_convolutional_change_detection).

## Citation
If you found our project helpful, please cite our paper:
```
@article{hong2023spectralgpt,
  title={SpectralGPT: Spectral foundation model},
  author={Hong, Danfeng and Zhang, Bing and Li, Xuyang and Li, Yuxuan and Li, Chenyu and Yao, Jing and Yokoya, Naoto and Li, Hao and Jia, Xiuping and Plaza, Antonio and others},
  journal={arXiv preprint arXiv:2311.07113},
  year={2023}
}
```
