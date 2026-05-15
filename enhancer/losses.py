import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #get vgg
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        #split it into features
        features = vgg.features
        self.feature_extractor = nn.Sequential(*list(features.children())[:9])
        #freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.criterion = nn.MSELoss()

    def forward(self, prediction, target):
        pred_normed = (prediction - self.mean) / self.std
        target_normed = (target - self.mean) / self.std
        
        pred_features = self.feature_extractor(pred_normed)
        with torch.no_grad():
            target_features = self.feature_extractor(target_normed)
        
        return self.criterion(pred_features, target_features)

