# test.py (corrected version using CustomImageFolder and updated model path)

import os
import torch
from torchvision.utils import save_image
from config import Config
from models import Generator
from data_loader import CustomImageFolder, get_transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_generator(model_path):
    generator = Generator().to(Config.DEVICE)
    generator.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    generator.eval()  # inference mode
    return generator

def load_test_images():
    transform = get_transforms()
    dataset = CustomImageFolder(os.path.join(Config.DATA_ROOT, "testB"), transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def test(model_path="generator_photo2vangogh.pth", save_folder="test_outputs"):
    os.makedirs(save_folder, exist_ok=True)
    generator = load_generator(model_path)
    test_loader = load_test_images()

    print("Generating Van Gogh-style images from real photos...")

    for i, real_img in enumerate(test_loader):
        real_img = real_img.to(Config.DEVICE)
        with torch.no_grad():
            fake_img = generator(real_img)

        # Un-normalize the image
        fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)

        # Save to disk
        save_path = os.path.join(save_folder, f"vangogh_{i:03d}.png")
        save_image(fake_img, save_path)

        # Optional: show
        img_np = fake_img.squeeze().permute(1, 2, 0).cpu().numpy()
        plt.imshow(img_np)
        plt.title(f"Generated Van Gogh Style Image {i}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    test()
