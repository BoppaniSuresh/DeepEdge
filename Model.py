import torch
import torch.nn as nn

# Model Architecture

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64*50*50, 512)
        self.fc2 = nn.Linear(512, 2)
    def forward(self, x):
        x =  self.conv1(x)
        #print('shape of x after conv1 : ', x.shape) #[32, 16, 50, 50]
        x = self.conv2(x)
        #print('shape of x after conv2 : ', x.shape) #[32, 32, 50, 50]
        x = self.conv3(x)
        #print('shape of x after conv3 : ', x.shape) #[32, 64, 50, 50]
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        #print('shape of x after denser layer 1 : ', x.shape) [32, 512]
        x = self.fc2(x)
        #print('shape of x after denser layer 2 : ', x.shape) [32, 2]
        
        return x 
    
