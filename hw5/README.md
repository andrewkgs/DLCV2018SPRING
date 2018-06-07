# DLCV homework 5: Action Recognition
> Course: [Deep Learning for Computer Vision (Spring 2018)](http://vllab.ee.ntu.edu.tw/dlcv.html)\
> Instructor: Prof. [Yu-Chiang Frank Wang](http://vllab.ee.ntu.edu.tw/members.html)


## Task Description
Train an RNN model to perform trimmed action recognition and temporal action segmentation in full-length videos.


## Toolkit requirements
[![Python version](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow version](https://img.shields.io/badge/TensorFlow-1.6.0-green.svg)](https://pypi.python.org/pypi/tensorflow/1.6.0)
[![Keras version](https://img.shields.io/badge/Keras-2.1.5-green.svg)](https://pypi.python.org/pypi/Keras/2.1.5)

## Execution Commands
* Reproduce the results in the report.
```sh
bash hw5_p1.sh <directory of trimmed validation videos folder> <path of ground-truth csv file> <directory of output labels folder>
bash hw5_p2.sh <directory of trimmed validation/test videos folder> <path of ground-truth csv file> <directory of output labels folder>
bash hw5_p3.sh <directory of full-length validation videos folder> <directory of output labels folder>
```
* Train
```
python3 p1.py train -tr=<train video directory> -v=<validation video directory> -trl=<train ground truth file> -vl=<validation ground truth file>
python3 p2.py train -tr=<train video directory> -v=<validation video directory> -trl=<train ground truth file> -vl=<validation ground truth file>
python3 p3.py train -tr=<train video directory> -v=<validation video directory> -trl=<train ground truth directory> -vl=<validation ground truth directory>
```

* Test
```
python p1.py test -te=<test video directory> -tel=<test ground truth file> -o=<output directory> -on=<output file name>
python p2.py test -te=<test video directory> -tel=<test ground truth file> -o=<output directory> -on=<output file name>
python p3.py test -te=<test video directory> -tel=<test ground truth directory> -o=<output directory>
```
