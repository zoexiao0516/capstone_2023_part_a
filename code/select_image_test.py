import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image



torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

image_path = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_500/'

IMAGE_DIM = 227


dataset = ImageFolder(root=image_path)
transform = transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor()])

print('Dataset created')
for img, label in dataset:
    print(img, label)
