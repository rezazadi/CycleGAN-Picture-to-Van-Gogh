# data_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Config

# Image transforms: resize → tensor → normalize
def get_transforms():
    return transforms.Compose([
        transforms.Resize((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# ✅ Custom dataset that loads images from a folder with NO class structure
class CustomImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# 🔄 Create PyTorch DataLoaders for trainA and trainB
def get_dataloaders():
    transform = get_transforms()
    dataset_A = CustomImageFolder(os.path.join(Config.DATA_ROOT, "trainA"), transform)
    dataset_B = CustomImageFolder(os.path.join(Config.DATA_ROOT, "trainB"), transform)

    loader_A = DataLoader(dataset_A, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    loader_B = DataLoader(dataset_B, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)

    return loader_A, loader_B
