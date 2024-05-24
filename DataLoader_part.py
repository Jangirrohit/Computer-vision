import torch
import numpy as np
import cv2
from sklearn.datasets import make_classification
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


# part1

class coustom_dataset:
    def __init__(self,data,targets):
        self.data=data
        self.targets=targets
    def __len__(self):
        return (len(self.data))
    def __getiteam__(self, idx):
        targets= self.targets[idx]
        data= self.data[idx]
        return{
            "targets": torch.tensor(targets),
            "data": torch.tensor(data)
        }
data, targets=make_classification(n_samples=1000)
dataset= coustom_dataset(data=data, targets=targets)
dataset_loder= torch.utils.data.DataLoader(dataset,batch_size=32, shuffle=True, num_workers=4)

# part2

class CustomDataset:
    def __init__(self,image_path,targets,transform):
        self.image_path=image_path
        self.targets=targets
        self.transform=transform
    def __len__(self):
        return (len(self.image_path))
    def __getitem__(self, idx):
        targets= self.targets[idx]
        image=cv2.imread(self.image_path[idx])
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image= image.transform(image)
        image=np.transpose(image, (2,0,1)).astype(np.float32)
        return{
            "targets": torch.tensor(targets),
            "image": torch.tensor(image)
        }
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.ToTensor()
])
data, targets=make_classification(n_samples=1000,n_features=784,n_classes=2)
data = data.reshape((n_samples, 28, 28))
# dataset= CustomDataset(data=data, targets=targets)
# dataset_loder= torch.utils.data.DataLoader(dataset,batch_size=32, shuffle=True, num_workers=4)


