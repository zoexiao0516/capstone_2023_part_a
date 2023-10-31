import torch
import torch.nn as nn
# import pickle

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

transform = transforms.Compose([
        transforms.ToTensor()
    ])

epoch = 42

image_path = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_500'
dataset = ImageFolder(root=image_path)

models_path = '/mnt/home/cchou/ceph/Capstone/models/'
with open(models_path+f'alexnet_states_e{epoch}.pkl', 'rb') as file:
    model = Net()
    model_loaded = torch.load(file)
    state_dict = model_loaded['model'] #skip this for '/models/models_150_b512' case
    model.load_state_dict(state_dict)    

layers_list = list(model.children())

for layer in range(len(layers_list)):
    print("Layer:", layer)
    clipped_model = nn.Sequential(*children_list[:i])
    for image, target in range(dataset):
        image = transform(image)
        image = image.unsqueeze(0)
        output = clipped_model(image)
        intermediate_output = intermediate_output.view(-1)

    current_epoch.append((current_radius, current_dimension))


#X should be a list of numpy arrays and each array will be (N * 500) projected on (3000 * 500)
epoch_layer_data = {'epoch':epoch, 'layer':layer, 'class': , X}


