# %%

from utils import test_loader
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report,cohen_kappa_score
import torch
import torchvision.models as models
import time 
import numpy as np

num_classes=5

resnet = models.resnet50(weights=None)
# Adapting the last layer to the 5-class classification task
num_features = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_features, num_classes)
#Load the weights
weights_path="models/resnet50_aptos19_best.pth"
state_dict = torch.load(weights_path, map_location="cuda")
resnet.load_state_dict(state_dict)

def evaluate(model,loader,device="cuda"):

    model.to(device)
    model.eval()

    y_pred=[]
    y_true=[]

    start_time=time.time()

    with torch.no_grad():
        for batch in loader:
            x,y=batch
            x=x.to(device)
            logits=model(x)
            preds=torch.argmax(logits,dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    finish_time=time.time()
    inference_time=finish_time-start_time

    metrics = {
            "accuracy": np.round(accuracy_score(y_true, y_pred),2),
            "f1_macro": np.round(f1_score(y_true, y_pred, average="macro"),2),
            "quadratic_kappa": np.round(cohen_kappa_score(y_true, y_pred, weights="quadratic"),2)
        }

    print(confusion_matrix(y_true,y_pred))
    print(f"Inference time on {device} : {inference_time:.2f} seconds.")
    print(f"Performace : {metrics}")

if __name__ == "__main__":
    evaluate(resnet,test_loader,device="cuda")
# %%
