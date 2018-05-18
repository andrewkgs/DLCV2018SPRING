# DLCV homework 4: Generative Models
> Course: [Deep Learning for Computer Vision (Spring 2018)](http://vllab.ee.ntu.edu.tw/dlcv.html)\
> Instructor: Prof. [Yu-Chiang Frank Wang](http://vllab.ee.ntu.edu.tw/members.html)


## Task Description
Generate images by VAE and GAN, and perform feature disentanglement by ACGAN.


## Toolkit requirements
[![Python version](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow version](https://img.shields.io/badge/TensorFlow-1.6.0-green.svg)](https://pypi.python.org/pypi/tensorflow/1.6.0)


## Execution Commands

### Reproduce the images in the report
```sh
bash hw4.sh <absolute path of hw4_data directory> <output directory>
```

### VAE
* Train model
```
python3 vae.py train -d <data path> -s <save model path>
```

* Reconstruct images
```
python3 vae.py reconstruct -d <data path> -o <output path> -l <load model path>
```

* Generate images
```
python3 vae.py generate -o <output path> -l <load model path>
```

### DCGAN
* Train model
```
python3 dcgan.py train -d <data path> -s <save model path>
```

* Generate images
```
python3 dcgan.py train -o <output path> -l <load model path>
```

### ACGAN
* Train model
```
python3 acgan.py train -d <data path> -s <save model path>
```

* Generate images
```
python3 acgan.py -o <output path> -l <load model path>
```
