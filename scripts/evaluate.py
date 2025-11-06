# %%
from scripts.utils import test_loader,get_teacher,get_student
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report,cohen_kappa_score
import torch
import torchvision.models as models
import time 
import numpy as np

def evaluate(model,loader,device="cuda"):

    model.to(device)
    model.eval()

    y_pred=[]
    y_true=[]

    print("Evaluation starting")
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
            "quadratic_kappa": np.round(cohen_kappa_score(y_true, y_pred, weights="quadratic"),2)
        }

    print(confusion_matrix(y_true,y_pred))
    print(f"Inference time on {device} : {inference_time:.2f} seconds.")
    print(f"Performace : {metrics}")

    return(metrics)

if __name__ == "__main__":

    weights_path_teacher="models/resnet50_aptos19_best.pth"
    teacher=get_teacher(weights_path_teacher,device="cpu")

    weights_path_student="models/mobilenetv3_distilled_best.pth"
    student=get_student(weights_path_student,device="cpu")

    evaluate(teacher,test_loader,device="cpu")
    evaluate(student,test_loader,device="cpu")
# %%
