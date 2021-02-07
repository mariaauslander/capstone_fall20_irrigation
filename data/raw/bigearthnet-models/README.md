# Deep Learning Models for BigEarthNet-S2 with 43 Classes
This repository contains code to use the [BigEarthNet](http://bigearth.net) Sentinel-2 (denoted as BigEarthNet-S2) archive with the original CORINE Land Cover (CLC) Level-3 class nomenclature of 43 classes for deep learning applications.

If you use the BigEarthNet-S2 archive or our pre-trained models, please cite the paper given below:

> G. Sumbul, M. Charfuelan, B. Demir, V. Markl, “[BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding](http://bigearth.net/static/documents/BigEarthNet_IGARSS_2019.pdf)”, IEEE International Geoscience and Remote Sensing Symposium, pp. 5901-5904, Yokohama, Japan, 2019.

```
@inproceedings{BigEarthNet,
    author = {Gencer Sumbul and Marcela Charfuelan and Begüm Demir and Volker Markl},
    title = {BigEarthNet: A Large-Scale Benchmark Archive For Remote Sensing Image Understanding},
    booktitle={IEEE International Geoscience and Remote Sensing Symposium}, 
    year = {2019},
    pages = {5901--5904}
    doi={10.1109/IGARSS.2019.8900532}, 
    month={July}
}
```

If you are interested in the nomenclature of 19 classes, check [here](https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-19-models).

## Pre-trained Deep Learning Models
We provide code and model weights for the following deep learning models that have been pre-trained on BigEarthNet-S2 with the original Level-3 class nomenclature of CLC 2018 (which includes 43 classes) for scene classification:


| Model Names   | Pre-Trained TensorFlow Models                                                                                                                               | 
| ------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| K-Branch CNN  | [K-BranchCNN.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-S2_43-Classes/K-BranchCNN.zip) |
| VGG16         | [VGG16.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-S2_43-Classes/VGG16.zip)            |
| VGG19         | [VGG19.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-S2_43-Classes/VGG19.zip)            |
| ResNet50      | [ResNet50.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-S2_43-Classes/ResNet50.zip)      |
| ResNet101     | [ResNet101.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-S2_43-Classes/ResNet101.zip)   | 
| ResNet152     | [ResNet152.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-S2_43-Classes/ResNet152.zip)   |

The results provided in the [BigEarthNet-S2 paper](http://bigearth.net/static/documents/BigEarthNet_IGARSS_2019.pdf) are different from those obtained by the models given above due to the selection of different train, validation and test sets.

The TensorFlow code for these models can be found [here](https://gitlab.tu-berlin.de/rsim/bigearthnet-models-tf).

The pre-trained models associated to other deep learning libraries will be released soon.

## Generation of Training/Test/Validation Splits
After downloading the raw images from https://www.bigearth.net, they need to be prepared for your ML application. We provide the script `prep_splits.py` for this purpose. It generates consumable data files (i.e., TFRecord) for training, validation and test splits which are suitable to use with TensorFlow. Suggested splits can be found with corresponding csv files under `splits` folder. The following command line arguments for `prep_splits.py` can be specified:

* `-r` or `--root_folder`: The root folder containing the raw images you have previously downloaded.
* `-o` or `--out_folder`: The output folder where the resulting files will be created.
* `-n` or `--splits`: A list of CSV files each of which contains the patch names of corresponding split.
* `-l` or `--library`: A flag to indicate for which ML library data files will be prepared: TensorFlow.

To run the script, either the GDAL or the rasterio package should be installed. The TensorFlow package should also be installed. The script is tested with Python 2.7, TensorFlow 1.3, and Ubuntu 16.04. 

**Note**: BigEarthNet-S2 patches with high density snow, cloud and cloud shadow are not included in the training, test and validation sets constructed by the provided scripts (see the list of patches with seasonal snow [here](http://bigearth.net/static/documents/patches_with_seasonal_snow.csv) and that of cloud and cloud shadow [here](http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv)). 


## Authors
[**Gencer Sümbül**](http://www.user.tu-berlin.de/gencersumbul/)

[**Tristan Kreuziger**](https://www.rsim.tu-berlin.de/menue/team/tristan_kreuziger/)


## License
The BigEarthNet-S2 Archive is licensed under the **Community Data License Agreement – Permissive, Version 1.0** ([Text](https://cdla.io/permissive-1-0/)).

The code in this repository to facilitate the use of the archive is licensed under the **MIT License**:

```
MIT License

Copyright (c) 2019 The BigEarthNet Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
