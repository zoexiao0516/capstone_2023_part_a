import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms


# Set the paths
save_path = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_50/'
# Ensure the save_path exists
os.makedirs(save_path, exist_ok=True)
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


# Define a function to save correctly labeled images
def save_correctly_labeled_images(dataset, model, save_path):
    model.eval()
    transform = transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for class_label, class_name in enumerate(dataset.classes):
            # Create a subdirectory for the class in the save_path
            class_save_dir = os.path.join(save_path, class_name)
            os.makedirs(class_save_dir, exist_ok=True)
            
            correct_count = 0
            total_count = 0

            for i in range(len(dataset.targets)):
                if dataset.targets[i] == class_label:
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
save_correctly_labeled_images(image_path, model, save_path)


            