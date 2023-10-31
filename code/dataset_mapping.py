from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


TRAIN_IMG_DIR = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_500'
IMAGE_DIM = 227


dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

print('Dataset created')
dataloader = data.DataLoader(
    dataset,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    batch_size=2)
print('Dataloader created')

class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Example usage of class to label mapping
print("Class to Label Mapping:")
for class_name, label in class_to_idx.items():
    print(f"{class_name}: {label}")