# Overview

This git repo consists of code for experimenting with various CNN architectures to analyze representations from different learning rules.

Env Set-up
```
conda create -n capstone_2023 python=3.10
conda activate capstone_2023
pip install -r requirements.txt
```
## Dataset

We are using the [ILSVRC 2012 dataset](http://www.image-net.org/challenges/LSVRC/2012/), also known as the 'ImageNet 2012 dataset'.
The data size is dreadfully large (138G!), but this amount of dataset is required for successful training of NN.

### Training images

There is a tar file for each synset, named by its WNID. The image files are named 
as x_y.JPEG, where x is WNID of the synset and y is an integer (not fixed width and not
necessarily consecutive). All images are in JPEG format.

There are a total of 1281167 images for training. The number of images for each 
synset ranges from 732 to 1300


### Validation images
*update later


### Testing images
*update later


The images vary in dimensions and resolution. Many applications resize / crop all of the images to 256x256 pixels.


ImageNet 2012's dataset structure is already arranged as `/root/[class]/[img_id].jpeg`, so using `torchvision.datasets.ImageFolder` is convenient.