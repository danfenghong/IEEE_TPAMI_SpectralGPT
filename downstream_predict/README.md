
### Downstream Pixel Tasks Prediction
For the segmentation and change detection prediction program, we recommend using the provided `predict.py` file, which will save the visualization results. The only tasks you need to complete are loading the pre-trained model and the image file to ensure correct inference.

```shell
cd /downstream_predict/OCSD
python predict.py
```

```shell
cd /downstream_predict/SegMunich
python predict.py
```

### Downstream Task Finetuned File

Please visit our link at [zenodo](https://zenodo.org/records/8412377) for the latest version. 

For the downstream remote sensing change detection on the OSCD dataset, please use the spectralGPT+_54_28.pth checkpoint.
For the downstream remote sensing semantic segmentation on the SegMunich dataset, please use the model_3con_loss_51_0.pth checkpoint.

Necessary modifications of the checkpoint name in the predict.py are needed.
