
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df=pd.read_csv(r"data/file_paths_labels.csv").drop(columns=["Unnamed: 0"])

train_idx,test_idx=train_test_split(np.arange(0,len(df)),random_state=0)

df_train=df.iloc[train_idx,:]
df_test=df.iloc[test_idx,:]

train_idx,val_idx=train_test_split(np.arange(0,len(df_train)))
df_train,df_val=df_train.iloc[train_idx,:],df_train.iloc[val_idx,:]

frequences=(df_train["label"].value_counts()/len(df_train)).to_dict()

# %%
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image

class eyes_dataset(Dataset):
    
    def __init__(self,df,transform=None):
        self.df=df
        self.transform=transform


    def __len__(self):
        return(len(self.df))
    
    def __getitem__(self,idx):
        x=Image.open(self.df["img_path"].iloc[idx])
        y=self.df["label"].iloc[idx]

        if self.transform:
            x = self.transform(x)

        return(x,y)
    
img_size=256
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),               
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


train_ds=eyes_dataset(df_train,transform=train_transforms)
train_loader=DataLoader(train_ds,batch_size=32,num_workers=3)

test_ds=eyes_dataset(df_test,transform=val_transforms)
test_loader=DataLoader(test_ds,batch_size=32,shuffle=False,num_workers=3)

val_ds=eyes_dataset(df_val,transform=val_transforms)
val_loader=DataLoader(val_ds,batch_size=32,shuffle=False,num_workers=3)

# %%
import torch
import torch.nn.functional as F
import torch.nn as nn 

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(1)  # pas de pondération spécifique
        else:
            # convertit la liste alpha en tenseur
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size] (entiers 0–num_classes-1)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
        pt = torch.exp(-ce_loss)
       
        if self.alpha.numel() > 1:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha.to(logits.device)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# %%
from utils import test_loader
from torchvision.models import resnet50,mobilenet_v3_small
import torch.nn as nn 
import torch 

def get_student(state_dict_path=None,device=None):
    mobile_net=mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1")

    mobile_net.classifier[3]=nn.Linear(1024,5)     

    
    if state_dict_path : 

        state_dict = torch.load(state_dict_path, map_location=device)
        mobile_net.load_state_dict(state_dict)

    return(mobile_net)

from torchvision.models import resnet50

def get_teacher(model_path=None,device="cuda"):

    teacher = resnet50(weights="IMAGENET1K_V1")
    teacher.fc = nn.Linear(2048, 5)

    if model_path : 
        teacher.load_state_dict(torch.load(model_path))
    else : 
        teacher = resnet50(weights="IMAGENET1K_V1")

    return(teacher)
# %%

# %% 


