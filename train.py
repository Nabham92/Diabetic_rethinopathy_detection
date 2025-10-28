# %%
from torchvision.models import resnet50
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn 
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score,confusion_matrix
import numpy as np
from utils import FocalLoss,train_loader,val_loader,df_train

# --- Initialisation du modèle ---
model = resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(2048, 5)

# Freeze tous les paramètres
for param in model.parameters():
    param.requires_grad = False

# Défreeze les couches hautes + la FC
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True


model = model.to("cuda")

# --- Hyperparamètres ---
N = 30

lr = 1e-4


optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

scheduler = CosineAnnealingLR(optimizer, T_max=N)

class_counts = df_train["label"].value_counts().sort_index()
frequences = class_counts / class_counts.sum()

alpha = (1.0 / frequences)
alpha = alpha / alpha.sum()  

criterion = FocalLoss(alpha=alpha.tolist(), gamma=2.0)




# --- Suivi des métriques ---
cks = []
lrs = []

# --- Early stopping ---
best_kappa = -1
patience = 5         # nombre d’epochs tolérées sans amélioration
patience_counter = 0
best_state = None

for epoch in range(N):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to("cuda"), y.to("cuda")
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 🔹 Mise à jour du scheduler après chaque epoch
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    lrs.append(current_lr)

    # --- Validation ---
    model.eval()
    val_labels, val_preds = [], []
    with torch.no_grad():
        for val_batch in val_loader:
            x_val, y_val = val_batch
            x_val = x_val.to("cuda")
            logits_val = model(x_val)
            preds = torch.argmax(logits_val, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(y_val.numpy())

    metrics = {
        "accuracy": accuracy_score(val_labels, val_preds),
        "f1_macro": f1_score(val_labels, val_preds, average="macro"),
        "quadratic_kappa": cohen_kappa_score(val_labels, val_preds, weights="quadratic")
    }
    print(confusion_matrix(val_labels,val_preds))
    ck = metrics["quadratic_kappa"]
    cks.append(ck)

    print(f"Epoch {epoch+1}/{N} | LR={current_lr:.6f} | "
          f"Val Acc={metrics['accuracy']:.3f} | F1={metrics['f1_macro']:.3f} | QWK={ck:.3f}")

    # --- Early stopping ---
    if ck > best_kappa:
        best_kappa = ck
        best_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹️  Early stopping déclenché à l’epoch {epoch+1}")
            break

print(f"\n Best Quadratic-Kappa : {best_kappa:.3f}")

# %%
if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), "resnet50_aptos19_best.pth")
# %%
