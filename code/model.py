import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import gc

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

NUM_EPOCHS = 30
BATCH_SIZE = 512

IMAGE_DIM = 227
NUM_CLASSES = 50

INPUT_ROOT_DIR = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_500'
TRAIN_IMG_DIR = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_500'
OUTPUT_DIR = '/mnt/home/cchou/ceph/Capstone/'
CHECKPOINT_DIR = OUTPUT_DIR + '/models/model_no_softmax'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(nn.Module):
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
        # self.softmax_final = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    clear_gpu_memory()

    seed = torch.initial_seed()

    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)

    print(alexnet)
    print('Model created')

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
        num_workers=4,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.001)
    print('Optimizer created')

    criterion = nn.CrossEntropyLoss()

    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        start_time = time.time()
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            output = alexnet(imgs)
            loss = criterion(output, classes)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

            total_loss += loss.item()
        end_time = time.time()

        with torch.no_grad():
            print("Epoch", epoch, "Average loss", total_loss / len(dataloader), "Time taken per epoch",
                  end_time - start_time)
            _, preds = torch.max(F.softmax(output, dim=-1), 1)
            accuracy = torch.sum(preds == classes) / BATCH_SIZE
            print("Accuracy", accuracy)

            # save checkpoints
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_states_e{}.pkl'.format(epoch + 1))
            torch.save(alexnet.state_dict(), checkpoint_path)
