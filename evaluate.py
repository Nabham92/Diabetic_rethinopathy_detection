# %%

from utils import test_loader
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report,cohen_kappa_score
import torch
import torchvision.models as models
import time 

num_classes=5

resnet = models.resnet50(weights=None)
# Adapting the last layer to the 5-class classification task
num_features = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_features, num_classes)
#Load the weights
state_dict = torch.load("resnet50_aptos19_best.pth", map_location="cuda")
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
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "quadratic_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic")
        }

    print(confusion_matrix(y_true,y_pred))
    print(f"Inference time on {device} : {inference_time} seconds.")
    print(f"Performace : {metrics}")

if __name__ == "__main__":
    evaluate(resnet,test_loader,device="cuda")



# %%
