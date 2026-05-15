import torch
from enhancer.losses import PerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = PerceptualLoss().to(device)

pred = torch.rand(2, 3, 96, 96, device=device, requires_grad=True)
target = torch.rand(2, 3, 96, 96, device=device)

loss = loss_fn(pred, target)
print("Loss value:", loss.item())
print("Loss requires grad:", loss.requires_grad)

# Same input should be zero
loss_zero = loss_fn(pred, pred)
print("Loss for identical inputs:", loss_zero.item())