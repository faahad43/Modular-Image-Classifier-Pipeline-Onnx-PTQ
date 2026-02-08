from torchvision.transforms import v2
import torch

imageNet_mean = [0.485, 0.456, 0.406]
imageNet_std = [0.229, 0.224, 0.225]

def get_transforms(img_size:int):
    train_tfms = v2.Compose([
        v2.ToImage(), #convert image into tensor
        v2.RandomResizedCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=imageNet_mean, std=imageNet_std)
    ])

    val_tfms = v2.Compose([
        v2.ToImage(), #convert image into tensor
        v2.Resize(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=imageNet_mean, std=imageNet_std)
    ])
    
    return train_tfms, val_tfms