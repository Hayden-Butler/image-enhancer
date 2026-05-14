import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.utils import save_image
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid

from enhancer.model import SRModel
from enhancer.dataset import SRDataset
from enhancer.model import SRResModel

def main():
    #hyperparameters
    SCALE = 4
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_EPOCHS_IN_TRAINING = 30

    #use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    #dataloader
    stl = STL10(root="data", split="test", download=True)
    sr_dataset = SRDataset(stl, scale=SCALE)
    loader = DataLoader(sr_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    #metrics
    psnr_model = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_model = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_bicubic = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_bicubic = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    

    
    #model setup
    model = SRResModel(scale=SCALE)
    model.load_state_dict(torch.load(f"checkpoints/resnet_epoch_{NUM_EPOCHS_IN_TRAINING}.pth"))
    model = model.to(device)
    model.eval()

    #testing loop
    with torch.no_grad():
        for lr_batch,hr_batch in tqdm(loader, desc="Evaluating"):
            #send the images to the device
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            #make the prediction
            prediction = model(lr_batch).clamp(0, 1)
            
            bicubic_pred = F.interpolate(
                lr_batch,
                scale_factor=SCALE,
                mode="bicubic",
                align_corners=False,
            )
            bicubic_pred = bicubic_pred.clamp(0, 1)


            #loss metrics
            psnr_model.update(prediction, hr_batch)
            ssim_model.update(prediction, hr_batch)
            psnr_bicubic.update(bicubic_pred, hr_batch)
            ssim_bicubic.update(bicubic_pred, hr_batch)

    print(f"Model    — PSNR: {psnr_model.compute().item():.2f} dB | SSIM: {ssim_model.compute().item():.4f}")
    print(f"Bicubic  — PSNR: {psnr_bicubic.compute().item():.2f} dB | SSIM: {ssim_bicubic.compute().item():.4f}")

    lr, hr = sr_dataset[1]
    lr_batched = lr.unsqueeze(0).to(device)
    hr_batched = hr.unsqueeze(0).to(device)

    lr_display = F.interpolate(lr_batched, scale_factor=SCALE, mode="nearest")
    bicubic = F.interpolate(lr_batched, scale_factor=SCALE,
                            mode="bicubic", align_corners=False).clamp(0, 1)
    
    with torch.no_grad():
        model_pred = model(lr_batched).clamp(0, 1)

    stacked = torch.cat([lr_display, bicubic, model_pred, hr_batched], dim=0)
    grid = make_grid(stacked, nrow=4)

    save_image(grid, "outputs/resnet_comparison.png")
    print("Saved comparison image to outputs/comparison.png")

if __name__ == "__main__":
    main()