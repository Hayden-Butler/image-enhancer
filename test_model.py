import torch
from enhancer.model import SRResModel

model = SRResModel(scale=4, channels=64, num_blocks=8)
dummy = torch.randn(2, 3, 24, 24)
out = model(dummy)
print("Input shape: ", dummy.shape)
print("Output shape:", out.shape)
print("Parameters:  ", sum(p.numel() for p in model.parameters()))