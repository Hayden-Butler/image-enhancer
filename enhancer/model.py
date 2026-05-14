import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = identity + out
        return out
        

class SRResModel(nn.Module):
    def __init__(self, scale = 4, channels=64, num_blocks = 8):
        super().__init__()
        self.scale = scale
        self.conv_head = nn.Conv2d(in_channels=3,out_channels=channels,kernel_size=3,padding=1)
        #make as many channels and layers as you put in
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.conv_body_end = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1)
        self.conv_upscale = nn.Conv2d(in_channels=channels,out_channels=(3*self.scale**2),kernel_size=3,padding=1)
        self.pixel_shuffle = nn.PixelShuffle(self.scale)
    def forward(self, x):
        x = self.conv_head(x)
        identity = x
        x = self.body(x)
        x = self.conv_body_end(x)
        x = x + identity
        x = self.conv_upscale(x)
        x = self.pixel_shuffle(x)
        return x


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
        
