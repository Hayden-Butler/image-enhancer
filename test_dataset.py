from enhancer.dataset import SRDataset
from torchvision.utils import save_image
from torchvision.datasets import STL10

stl = STL10(root="data", split="train", download=True)
idx = 0
sr_dataset = SRDataset(base_dataset=stl, scale=4)

lr_tensor, hr_tensor = sr_dataset[0]

# Inspect
print("LR shape:", lr_tensor.shape, "min:", lr_tensor.min().item(), "max:", lr_tensor.max().item())
print("HR shape:", hr_tensor.shape, "min:", hr_tensor.min().item(), "max:", hr_tensor.max().item())
print("Dataset length:", len(sr_dataset))

save_image(lr_tensor, "outputs/sample_lr.png")
save_image(hr_tensor, "outputs/sample_hr.png")