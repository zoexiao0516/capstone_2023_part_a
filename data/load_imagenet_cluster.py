import os
import json
import random
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import shutil

# Load the class index file
with open('/scratch/yx1750/capstone2023/data/imagenet_class_index.json', 'r') as f:
    class_idx = json.load(f)

# Creat a dict mapping WNID to class label
id2class = {class_idx[str(k)][0]:class_idx[str(k)][1] for k in range(len(class_idx))}

# Creat a dict mapping label to index
class2id = {class_idx[str(k)][1]:class_idx[str(k)][0] for k in range(len(class_idx))}

# Get all class
def getAllNames():
    allClassNames = []
    for key in id2class:
        name = id2class[key]
        allClassNames.append(name)
    return allClassNames

allClassNames = getAllNames()

# Get random class
def randomSample(labels, seed, n):
    random.seed(seed)
    result = random.sample(labels, n)
    return result

random50 = randomSample(allClassNames, 3108, 50)

# Get random class WNID
def getId(labels):
    idList = []
    for each in labels:
        classId = class2id[each]
        idList.append(classId)
    return idList

random50_id = getId(random50)

# Set the data directory
data_dir = '/mnt/home/cchou/ceph/Data/imagenet/ILSVRC/Data/CLS-LOC/'

# Define data transforms for preprocessing (adjust as needed)
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256 pixels
    transforms.ToTensor(),          # Convert to tensor
])

ds = ImageNet(root=data_dir, split='train', transform=data_transform)

# Define the directory where you want to create your custom subset
custom_subset_root = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_500'

selected_classes = random50_id

# Define the number of images to keep (in this case, 500)
num_images_to_keep = 500

# Create the custom subset directory structure and get the first 5 images
for class_name in selected_classes:
    source_dir = os.path.join(data_dir, 'train', class_name)  # Assuming you're working with the training data
    target_dir = os.path.join(custom_subset_root, class_name)
    os.makedirs(target_dir, exist_ok=True)

    # List all the image files in the source directory
    image_files = os.listdir(source_dir)
    
    # Get the first 5 images and copy them to the target directory
    for filename in image_files[:num_images_to_keep]:
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)
        shutil.copy(source_file, target_file)  # You can use shutil.move() to move the files if you prefer

# Your custom subset with the first 5 images from each of the selected 50 classes is now ready
