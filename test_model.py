import torch
from enhancer.model import SRModel

model = SRModel(scale=4)
dummy_input = torch.randn(2, 3, 24, 24)
output = model(dummy_input)
print(dummy_input.shape)
print(output.shape)
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))