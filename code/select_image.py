import os
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Set the paths
save_path = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_50/'
image_path = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_50/'
models_path = '/mnt/home/cchou/ceph/Capstone/models/'
IMAGE_DIM = 227
epoch = 90

class Net(nn.Module):
    def __init__(self, num_classes=50):
        # Conv layers
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 11, stride = 2, padding = 5),
                        nn.BatchNorm2d(64),
                        nn.ReLU(), nn.MaxPool2d(kernel_size = 3, stride = 2))
 
        self.conv2 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(128),
                        nn.ReLU(), nn.MaxPool2d(kernel_size = 2))
        
        self.conv3 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(), nn.MaxPool2d(kernel_size = 2))

        
        self.conv4 = nn.Sequential(
                        nn.Conv2d(256, 512, kernel_size = 3),
                        nn.BatchNorm2d(512))
        

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=(512 * 25), out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes),
        )
        self.softmax_final = nn.Softmax(dim=-1)


with open(models_path+f'alexnet_states_e{epoch}.pkl', 'rb') as file:
    model = Net()
    model_loaded = torch.load(file)
    state_dict = model_loaded['model'] # skip this for '/models/models_150_b512' case
    model.load_state_dict(state_dict)


# Define the data transformations, including resizing
data_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize the images to a specific size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Create a dataset from your directory structure
dataset = ImageFolder(root=image_path, transform=data_transform)

# Create a DataLoader to iterate through the dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Set batch_size to 1 for one image at a time

model.eval()  # Set the model to evaluation mode

for class_idx in range(len(dataset.classes)):
    class_name = dataset.classes[class_idx]

    # Create a directory for each class in the output directory
    class_output_dir = os.path.join(save_path, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    class_images = []

    for image, label in dataloader:
        output = model(image)
        _, predicted_class = output.max(1)

        if predicted_class.item() == label.item():
            class_images.append((dataset.samples[dataloader.batch_sampler.sampler.data_source.data[label.item()]], output[0, label.item()].item()))

    # Sort the images by prediction confidence
    class_images.sort(key=lambda x: x[1], reverse=True)

    # Save the top 50 correctly classified images for this class
    for i, (image_path, confidence) in enumerate(class_images[:50], start=1):
        image_filename = os.path.basename(image_path[0])
        output_path = os.path.join(class_output_dir, f"top_{i}_{confidence:.4f}_{image_filename}")
        shutil.copy(image_path[0], output_path)

            