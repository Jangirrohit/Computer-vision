# part1

import torch 
import numpy as np
import cv2
from sklearn.datasets import make_classification
import torchvision.transforms as transforms

class coustom_dataset:
    def __init__(self, image_path, targets):
        self.image_path= image_path
        self.targets=targets

    def __len__(self):
        return(len(self.image_path))
    def __getitem__(self, idx):
        targets= self.targets[idx]
        image= cv2.imread(self.image_path[idx])
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=np.transpose(image, (2,0,1)).astype(np.float32)
        return{
            'image': torch.tensor(image),
            'targets':torch.tensor(targets)
        }



# part2


class coustom_dataset:
    def __init__(self, image_path, targets,data_transform):
        self.image_path= image_path
        self.targets=targets
        self.data_transform=data_transform

    def __len__(self):
        return(len(self.image_path))
    def __getitem__(self, idx):
        targets= self.targets[idx]
        image= cv2.imread(self.image_path[idx])
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image= self.data_transform(image)
        image=np.transpose(image, (2,0,1)).astype(np.float32)
        return{
            'image': torch.tensor(image),
            'targets':torch.tensor(targets)
        }
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

 

#  part3

# part4

class coustom_dataset:
    def __init__(self, image_path, targets):
        self.image_path= image_path
        self.targets=targets
    def __len__(self):
        return(len(self.image_path))
    def __getitem__(self, idx):
        targets= self.targets[idx]
        image= cv2.imread(self.image_path[idx],cv2.IMREAD_GRAYSCALE)
        image= cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=np.transpose(image, (2,0,1)).astype(np.float32)
        return{
            'image': torch.tensor(image),
            'targets':torch.tensor(targets)
        }
    
# part5.





