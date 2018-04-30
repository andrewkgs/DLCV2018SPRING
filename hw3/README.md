# DLCV homework 3: Semantic Segmentation
> Course: [Deep Learning for Computer Vision (Spring 2018)](http://vllab.ee.ntu.edu.tw/dlcv.html)\
> Instructor: Prof. [Yu-Chiang Frank Wang](http://vllab.ee.ntu.edu.tw/members.html)


## Task Description
Perform semantic segmentation which predicts a label to each pixel on the aerial photograph dataset.


## Toolkit requirements
[![Python version](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow version](https://img.shields.io/badge/TensorFlow-1.6.0-green.svg)](https://pypi.python.org/pypi/tensorflow/1.6.0)
[![Keras version](https://img.shields.io/badge/Keras-2.1.5-green.svg)](https://pypi.python.org/pypi/Keras/2.1.5)


## Execution Commands

### Training
* For baseline model (VGG16-FCN32s):
```
python3 semantic_segmentation.py VGG16-FCN32s --train_data_dir=<training images directory> --valid_data_dir=<validation images directory> --batch_size=<batch size> --epochs=<training epochs> --model_dir=<saved model directory>
```

* For improved model (VGG16-FCN16s):
```
python3 semantic_segmentation.py VGG16-FCN16s --train_data_dir=<training images directory> --valid_data_dir=<validation images directory> --batch_size=<batch size> --epochs=<training epochs> --model_dir=<saved model directory>
```

### Testing
* For baseline model (VGG16-FCN32s):
```sh
bash hw3.sh <testing images directory> <output images directory>
```

* For improved model (VGG16-FCN16s):
```sh
bash hw3_best.sh <testing images directory> <output images directory>
```
