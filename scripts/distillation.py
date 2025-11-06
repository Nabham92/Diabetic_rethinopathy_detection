# %%
from dataclasses import dataclass
from scripts.evaluate import evaluate
from scripts.utils import train_loader, val_loader, test_loader, FocalLoss, get_student, df_train,set_seed
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy


@dataclass
class TrainConfig:
    n_epochs: int 
    lr: float
    loss: nn.Module
    device: str = "cuda"
    patience: int = 5  
    delta: float = 1e-4  

def train(model, loader, optimizer, train_config, val_loader=None, scheduler=None): 
    model.to(train_config.device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_kappa = -float("inf")

    epochs_no_improve = 0

    for epoch in range(train_config.n_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(loader):
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
        
        # 🔹 Évaluation et early stopping
        if val_loader:
            model.eval()
            with torch.no_grad():
                val_kappa = evaluate(model, val_loader, train_config.device)["quadratic_kappa"]
            
            print(f"Validation Kappa: {val_kappa:.4f}")

            # Vérifie si amélioration
            if val_kappa > best_kappa + train_config.delta:
                best_kappa = val_kappa
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                print(f"🔹 Nouveau meilleur modèle (Kappa={best_kappa:.4f})")
            else:
                epochs_no_improve += 1
                print(f" Pas d'amélioration du Kappa depuis {epochs_no_improve} epochs.")

            # 🔚 Early stopping
            if epochs_no_improve >= train_config.patience:
                print(f"⏹️ Early stopping déclenché à l'epoch {epoch+1}.")
                break

    # Recharge le meilleur modèle
    model.load_state_dict(best_model_wts)
    print(f"✅ Entraînement terminé. Meilleur Kappa: {best_kappa:.4f}")

    return model

set_seed(1)
class_counts = df_train["label"].value_counts().sort_index()
frequences = class_counts / class_counts.sum()
alpha = (1.0 / frequences)
alpha = alpha / alpha.sum()

criterion = FocalLoss(alpha=alpha.tolist(), gamma=2.0)
train_config = TrainConfig(
    n_epochs=30,
    lr=1e-4,
    loss=criterion,
    patience=5,    
    delta=1e-4
)

mobile_net = get_student(device="cuda")

# Geler les blocs initiaux

for name, param in mobile_net.features.named_parameters():
    bloc_idx = int(name.split(".")[0])
    if bloc_idx <= 6: 
        param.requires_grad = False

optimizer = Adam(
    filter(lambda p: p.requires_grad, mobile_net.parameters()), 
    lr=train_config.lr
)

scheduler = CosineAnnealingLR(optimizer, T_max=train_config.n_epochs)

# Entraînement avec early stopping
trained_model = train(mobile_net, train_loader, optimizer, train_config, val_loader=val_loader, scheduler=scheduler)

# %%

evaluate(mobile_net,test_loader,device="cpu")
# %%
