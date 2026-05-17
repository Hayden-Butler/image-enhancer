import torch
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from enhancer.model import SRResModel
#run with this python infer.py inputs/cat.jpg  

def main():
    #arguement parser
    parser = argparse.ArgumentParser(description="Upscale an image 4x using a trained SR model")
    parser.add_argument("input", help="Path to input image (e.g. inputs/bulbasaur.jpg)")
    parser.add_argument("--output", default="outputs/upscaled.png",
                        help="Where to save the upscaled image")
    args = parser.parse_args()

    #parameters
    SCALE = 4
    CHECKPOINT_PATH = "checkpoints/resnet_perceptual_epoch_30.pth" #used checkpoint, change it if you run it for longer

    #convert image to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img = Image.open(args.input).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    print(f"Input shape: {tuple(tensor.shape)}")

    #model
    model = SRResModel(scale=SCALE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(tensor).clamp(0, 1)

    output = output.squeeze(0)
    save_image(output, args.output)

if __name__ == "__main__":
    main()