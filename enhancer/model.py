import torch.nn as nn

class SRModel(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        #relu is the non linear function
        self.relu = nn.ReLU()
        #convelution layers
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=(3*self.scale**2),kernel_size=3,padding=1)
        #rearange the 48 pixels into 96x3
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        return x
        
