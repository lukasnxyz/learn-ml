import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from going_modular.going_modular import data_setup, engine

import os
import zipfile
from pathlib import Path
import requests

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    data_path = Path("data/")
    image_path = data_path/"pizza_steak_sushi"
    
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        with open(data_path/"pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)
        with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...")
            zip_ref.extractall(image_path)
        os.remove(data_path/"pizza_steak_sushi.zip")
    
    train_dir = image_path/"train"
    test_dir = image_path/"test"
    
    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    auto_transforms = weights.transforms()
    train_dloader, test_dloader, class_names = data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=auto_transforms,
            batch_size=32
    )

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
