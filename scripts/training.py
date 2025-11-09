# %%
from dataclasses import dataclass
from scripts.evaluate import evaluate
from scripts.utils import train_loader, val_loader, FocalLoss, df_train,set_seed,get_teacher
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from scripts.evaluate import evaluate

@dataclass
class TrainConfig:
    n_epochs: int 
    lr: float
    loss: nn.Module
    device: str = "cuda"
    patience: int = 5  
    delta: float = 1e-3

def train(model, loader, optimizer, train_config, val_loader=None, scheduler=None): 
    model.to(train_config.device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_kappa = -float("inf")

    epochs_no_improve = 0

    for epoch in range(train_config.n_epochs):
        model.train()
        epoch_loss = 0.0
        
        for _, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            x, y = x.to(train_config.device), y.to(train_config.device)
            
            y_pred = model(x)
            batch_loss = train_config.loss(y_pred, y)
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
        
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{train_config.n_epochs} --- Training Loss : {epoch_loss:.4f}")
        
        # Validation 
        if val_loader:
            model.eval()
            with torch.no_grad():
                val_kappa = evaluate(model, val_loader, train_config.device)["quadratic_kappa"]
            
            print(f"Validation Kappa: {val_kappa:.4f}")

            # Improvement check
            if val_kappa > best_kappa + train_config.delta:
                best_kappa = val_kappa
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                print(f" New best_model (Kappa={best_kappa:.4f})")
            else:
                epochs_no_improve += 1
                print(f" No Kappa improvement since {epochs_no_improve} epochs.")

            # Early stopping
            if epochs_no_improve >= train_config.patience:
                print(f" Early triggered  at {epoch+1}.")
                break

    # Loading the best model 
    model.load_state_dict(best_model_wts)
    print(f" Training finished. Best Kappa: {best_kappa:.4f}")

    return model

if __name__=="__main__" : 

    from torchvision.models import resnet50

    set_seed(1)

    # ResNet50 pretrained on ImageNet
    resnet = resnet50(weights="IMAGENET1K_V1")
    resnet.fc = nn.Linear(2048, 5)

    # Freeze all layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Unfreezing layers to train
    for param in resnet.layer3.parameters():
        param.requires_grad = True
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    for param in resnet.fc.parameters():
        param.requires_grad = True

    # Defining weights to prenalize errors on minority classes during training 
    class_counts = df_train["label"].value_counts().sort_index()
    frequences = class_counts / class_counts.sum()
    alpha = (1.0 / frequences)
    alpha = alpha / alpha.sum()

    criterion = FocalLoss(alpha=alpha.tolist(), gamma=2.0)

    # Training 
    train_config = TrainConfig(
        n_epochs=30,
        lr=1e-4,
        loss=criterion,
        patience=5,    
        delta=1e-4)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, resnet.parameters()), 
        lr=train_config.lr
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.n_epochs)

    # Entraînement avec early stopping
    trained_model = train(resnet, train_loader, optimizer, train_config, val_loader=val_loader, scheduler=scheduler)

    torch.save(trained_model.state_dict(), "backend/models/resnet50_aptos19_best.pth")

