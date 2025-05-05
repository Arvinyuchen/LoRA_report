import torch
import os
from torchvision.datasets import CIFAR100, collate_fn
from torch.utils.data import DataLoader, dataset
import dataset
from model import inject_lora_into_clip

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
    train_ds = dataset.ContrastiveCIFAR(cifar_train)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2)

    
    train_loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    # add lora
    inject_lora_into_clip()
    
    freeze_clip(model)
    for name, parameters in 

    model.to(device)

    # set optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # train
    train(model, train_loader, preprocess, device, optimizer, 200, CFG.model_path)
