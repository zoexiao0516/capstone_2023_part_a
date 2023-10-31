import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os


IMAGE_PATH = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_500'
MODELS_PATH = '/mnt/home/cchou/ceph/Capstone/models/model_no_softmax'
OUT_DIR = '/mnt/home/cchou/ceph/Capstone/Dataframes/'
# Ensure the save_path exists
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_DIM = 227
DIM = 3000

class Net(nn.Module):
    def __init__(self, num_classes=50):
        # Conv layers
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512))

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=(512 * 25), out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


epoch = 42
rows = []

dataset = datasets.ImageFolder(IMAGE_PATH, transforms.Compose([transforms.CenterCrop(IMAGE_DIM),
                                                               transforms.ToTensor()]))


with open(MODELS_PATH + f'model_states_e{epoch}.pkl', 'rb') as file:
    model = Net()
    model_loaded = torch.load(file)
    state_dict = model_loaded['model']
    model.load_state_dict(state_dict)

print("Model Loaded")
layers_list = list(model.children())

for layer in range(len(layers_list)):
    print("Layer:", layer)
    clipped_model = nn.Sequential(*layers_list[:i])
    X = []

    for i in range(0, len(dataset), 500):
        image, target = dataset[i:500]
        output = clipped_model(image)
        X_i = output.view(-1, image.shape[0])
        X.append(X_i.detach().to('cpu').numpy())

    dim_representation = X[0].shape[0]
    if dim_representation > DIM:
        M = np.random.randn(DIM, dim_representation)
        M /= np.sqrt(np.sum(M * M, axis=1, keepdims=True))
        # X_projected should be a list of numpy arrays and each array will be (N * 500) projected on (3000 * 500)
        X_projected = [np.matmul(M, X_i) for X_i in X]

    epoch_layer_data = {'epoch': epoch, 'layer': layer, 'X_projected': X_projected}
    rows.append(epoch_layer_data)

df = pd.DataFrame.from_records(rows)
print(df.head(5))
df.to_csv(OUT_DIR+str(epoch))