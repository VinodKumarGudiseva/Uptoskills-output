import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TaskImageDataset(Dataset):
    def __init__(self, task_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(task_dir, "**", "*.jpg"), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        
        # Simulating activity classification index (0-5)
        pseudo_label = hash(os.path.dirname(path)) % 6

        if self.transform:
            image = self.transform(image)

        return image, pseudo_label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def build_dataloader(task_dir, batch_size=16, is_train=True):
    train_tf, val_tf = get_transforms()
    tf = train_tf if is_train else val_tf
    dataset = TaskImageDataset(task_dir, transform=tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2, pin_memory=True)
    return loader