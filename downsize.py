from PIL import Image
imgName = "cat.jpg"
img = Image.open(f"inputs/{imgName}").convert("RGB")
# Downscale to 64x64 (will become 256x256 after 4x upscaling)
small = img.resize((64, 64), Image.BICUBIC)
small.save(f"inputs/{imgName}")