import os
import clip 
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as T



# -------------- Dataset ------------
class ContrastiveCIFAR(torch.utils.data.Dataset):
    def __init__(self, cifar_ds, template="a photo of a {}"):
        self.images  = cifar_ds.data
        self.labels  = cifar_ds.targets       
        self.classes = cifar_ds.classes        
        self.template = template
        self.prep = T.Compose([
            T.ToPILImage(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.prep(self.images[idx])
        caption = self.template.format(self.classes[self.labels[idx]])
        caption = caption.replace('_', ' ')
        return img, caption 



# ---------- Quick sampler / visualiser ----------
def sample_dataset(dataloader, n=4):
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(dataloader)
    images, captions = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{captions[j]:5s}' for j in range(n)))


def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images)
    return images, list(texts)



def main ():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    root = os.path.expanduser("~/.cache")
    cifar_train = CIFAR100(root, download=True, train=True)
    train_ds = ContrastiveCIFAR(cifar_train)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)

    sample_dataset(train_loader)


if __name__ == "__main__":
    main()

