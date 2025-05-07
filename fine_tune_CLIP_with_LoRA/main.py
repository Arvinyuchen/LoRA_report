import torch
import os
import torchvision
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, dataset
from dataset import collate_fn, ContrastiveCIFAR
from model import inject_lora_into_clip
from torch.optim import AdamW
from train import train, eval

import clip 


def ft_clip_lora():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)
    model = model.float()
    
    # Prepare datasets with CLIP preprocessing
    root = os.path.expanduser("~/.cache")
    cifar_train_raw = CIFAR100(root, download=True, train=True, transform=preprocess)
    cifar_test_raw  = CIFAR100(root, download=True, train=False, transform=preprocess)

    # Wrap in ContrastiveCIFAR to get image-text pairs
    train_ds = ContrastiveCIFAR(cifar_train_raw)
    test_ds  = ContrastiveCIFAR(cifar_test_raw)

    # Create DataLoaders
    batch_size = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    
    # Initial evaluation with frozen model
    # print("Evaluating zero-shot performance...")
    # eval(model, cifar_train_raw, cifar_test_raw, device)  # use raw dataset for image-only encoding
    
    # Inject LoRA
    inject_lora_into_clip(model, r=4, alpha=8)
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Train the model
    model_path = './cifar_clip_lora.pth'
    train(model, train_loader, preprocess, device, optimizer, 20, model_path)

    # Final evaluation after fine-tuning
    print("Evaluating after fine-tuning...")
    eval(model, cifar_train_raw, cifar_test_raw, device)


if __name__ == "__main__":
    ft_clip_lora()

