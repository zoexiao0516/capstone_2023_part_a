## Data Set

We are going to use the [ILSVRC 2012 dataset](http://www.image-net.org/challenges/LSVRC/2012/), also known as the 'ImageNet 2012 dataset'.
The data size is dreadfully large (138G!), but this amount of large-sized dataset is required for successful training of NN.

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