from torchvision import datasets
from torch.utils.data import DataLoader
from transform import get_transforms

Datasets = {
    "cifar10": {
        "cls": datasets.CIFAR10,
        "num_classes": 10,
        "train_args": {"train": True},
        "val_args": {"train": False}
    },
    "cifar100": {
        "cls": datasets.CIFAR100,
        "num_classes": 100,
        "train_args": { "train": True },
        "val_args" : { "train": False}   
    },
    "stl10": {
        "cls": datasets.STL10,
        "num_classes": 10,
        "train_args": { "split": "train"},
        "val_args": { "split": "test"}
    }
}

def dataset_loader(dataset='cifar10', path="data", batch_size = 32, num_worker=2, pin_memory=False, image_size: int = 224):
    
    if dataset not in Datasets:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_tfms, val_tfms = get_transforms(image_size)
    
    cfg = Datasets[dataset]
     
    train_ds = cfg["cls"](
        root=path,
        **cfg["train_args"], 
        download=True, 
        transform=train_tfms)
    
    val_ds = cfg["cls"](
        root=path,
        **cfg["val_args"], 
        download=True,
        transform=val_tfms
        )
       
    
        
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=True,
        pin_memory=pin_memory
        )
    
    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=False,
        pin_memory=pin_memory
        )
    
    return train_dataloader, val_dataloader, cfg["num_classes"]