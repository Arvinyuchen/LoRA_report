import torch
import os
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, dataset
from dataset import collate_fn, ContrastiveCIFAR
from model import inject_lora_into_clip
from torch.optim import AdamW
from train import train

import clip 


def ft_clip_lora():

    batch_size = 4

    # load model and preprocessing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)
    model = model.float()
    
    # load dataset and dataloader
    root = os.path.expanduser("~/.cache")
    cifar_train = CIFAR100(root, download=True, train=True)
    train_ds = ContrastiveCIFAR(cifar_train)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # add lora
    inject_lora_into_clip(model, r=4, alpha=8)
    
    # freeze parameters except lora
    for name, param in model.named_parameters():
        if 'lora_attn' in name or 'c_fc' in name or 'c_proj' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    model.to(device)

    # set optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # train
    model_path = './cifar_clip_lora.pth'
    train(model, train_loader, preprocess, device, optimizer, 200, model_path)


if __name__ == "__main__":
    ft_clip_lora()

