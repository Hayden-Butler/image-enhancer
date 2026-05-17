# Image Enhancer

A 4× image upscaler I built to learn PyTorch. Trained on STL-10.

## Results

Evaluated on STL-10 test set:

| Model | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic (no learning) | 21.52 | 0.6247 |
| My model (residual + perceptual loss) | 22.64 | 0.6714 |

Works decently on photos. Falls over on cartoons/sprite art because it was only trained on photographs.

## Run it

```bash
pip install -r requirements.txt
python infer.py path/to/image.jpg
```

## Dependencies

torch, torchvision, torchmetrics, tqdm, pillow
