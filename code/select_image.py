import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder


torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Set the paths
save_path = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_50/'
# Ensure the save_path exists
os.makedirs(save_path, exist_ok=True)
image_path = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_500/'
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
        # self.softmax_final = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.shape[0], -1)
        # print("Flattented shape", x.shape)
        x = self.classifier(x)
        # x = self.softmax_final(x)
        return x

with open(models_path+f'model_states_e{epoch}.pkl', 'rb') as file:
    model = Net()
    model_loaded = torch.load(file)
    # state_dict = model_loaded['model']
    model.load_state_dict(model_loaded)


# Create a dataset from your directory structure
dataset = ImageFolder(root=image_path)

# Define a function to save correctly labeled images
def save_correctly_labeled_images(dataset, model, save_path):
    model.eval()
    transform = transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor()
    ])

    with torch.no_grad():
        for class_idx, class_name in enumerate(dataset.classes):
            # Create a subdirectory for the class in the save_path
            class_save_dir = os.path.join(save_path, class_name)
            os.makedirs(class_save_dir, exist_ok=True)
            
            correct_count = 0
            total_count = 0

            for i in range(len(dataset.targets)):
                if dataset.targets[i] == class_idx:
                    image, target = dataset[i]
                    image = transform(image)
                    image = image.unsqueeze(0)  # Add batch dimension
                    output = model(image)
                    prediction = output.argmax(dim=1)

                    if prediction == target:
                        # Save the correctly classified image
                        image = transforms.ToPILImage()(image.squeeze(0))
                        image_path = os.path.join(class_save_dir, f"image_{correct_count}.jpg")
                        image.save(image_path)

                        correct_count += 1
                        if correct_count == 50:  # Save 50 images for each class
                            break

                    total_count += 1
                    if total_count == 500:  # Limit the number of images to 500 per class
                        break

# Call the function to save correctly labeled images
save_correctly_labeled_images(dataset, model, save_path)