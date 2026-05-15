import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.utils import save_image
from tqdm import tqdm

from enhancer.model import SRModel
from enhancer.dataset import SRDataset
from enhancer.model import SRResModel
from enhancer.losses import PerceptualLoss

def main():
    #hyperparameters
    SCALE = 4
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LR = 1e-3
    NUM_WORKERS = 4
    PERCEPTUAL_WEIGHT = 0.02

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    #dataloader
    stl = STL10(root="data", split="train", download=True)
    sr_dataset = SRDataset(stl, scale=4)
    loader = DataLoader(sr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    #model setup
    model = SRResModel(scale=SCALE).to(device)
    mse_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss().to(device)
    optimiser = torch.optim.Adam(model.parameters(),lr=LR)

    #training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        #progress bars
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        #compute loss and backpropogation for the batches
        for lr_batch,hr_batch in progress:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            prediction = model(lr_batch)
            pixel = mse_loss(prediction, hr_batch)
            perceptual = perceptual_loss(prediction, hr_batch)
            loss = pixel + PERCEPTUAL_WEIGHT * perceptual

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(
                pix=f"{pixel.item():.4f}",
                perc=f"{perceptual.item():.4f}",
                total=f"{loss.item():.4f}"
            )
            
        avg_loss = total_loss/num_batches
        
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        #save the state and images
        torch.save(model.state_dict(), f"checkpoints/resnet_perceptual_epoch_{epoch+1}.pth")
        save_sample(model, sr_dataset, device, epoch+1)

def save_sample(model, dataset, device, epoch):
    #save the model
    if epoch % 5 != 0:
        return
    model.eval()
    with torch.no_grad():
        lr,hr = dataset[0]
        lr_batched = lr.unsqueeze(0).to(device)
        pred = model(lr_batched).squeeze(0).cpu().clamp(0, 1)
        save_image(pred, f"outputs/resnet_perceptual_epoch_{epoch}_pred.png")
    model.train()
        

if __name__ == "__main__":
    main()


