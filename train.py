import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.utils import save_image
from tqdm import tqdm

from enhancer.model import SRModel
from enhancer.dataset import SRDataset

def main():
    SCALE = 4
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 1e-3
    NUM_WORKERS = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    stl = STL10(root="\data", split="train", download=True)
    sr_dataset = SRDataset(stl, scale=4)
    loader = DataLoader(sr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    model=SRModel(scale=SCALE).to(device)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(),lr=LR)
    

