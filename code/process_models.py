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
        self.softmax_final = nn.Softmax(dim=-1)

all_epochs_data = {}

epoch = 42
image_path = '/mnt/home/cchou/ceph/Data/imagenet_subset_50_500/n021076'
input_image =  # dimension should be 227
models_path = '/mnt/home/cchou/ceph/Capstone/models/'
with open(models_path+f'alexnet_states_e{epoch}.pkl', 'rb') as file:
    model = Net()
    model_loaded = torch.load(file)
    state_dict = model_loaded['model'] #skip this for '/models/models_150_b512' case
    model.load_state_dict(state_dict)

children_list = list(model.children())
current_epoch = []
for i in range(len(children_list)):
    print("Layer", i)
    clipped = nn.Sequential(*children_list[:i])
    intermediate_output = clipped(input_image)
    intermediate_output = intermediate_output.view(-1)
    #calculate radius and dimension of intermediate_output and add to you list/dicts
    # current_radius, current_dimension = #
    current_epoch.append((current_radius, current_dimension))

all_epochs_data[epoch] = current_epoch

#get the image in the relevant shape here
# output = model(input_image)


