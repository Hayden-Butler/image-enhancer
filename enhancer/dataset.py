import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SRDataset(Dataset):
    def __init__(self, base_dataset, scale=4):
        #dataset, scale and the tensor method
        self.base = base_dataset
        self.scale = scale
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        #get the high res image at the index
        hr_image, _ = self.base[idx]
        #find low res size
        hr_width, hr_height = hr_image.size
        lr_width, lr_height = hr_width//self.scale, hr_height//self.scale
        #resize hight res image
        lr_image = hr_image.resize((lr_width, lr_height), Image.Resampling.BICUBIC)
        #convert to tensor
        hr_tensor = self.to_tensor(hr_image)
        lr_tensor = self.to_tensor(lr_image)

        return lr_tensor, hr_tensor
