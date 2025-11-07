import torch
from PIL import Image
from models.mobile_net import get_student
import numpy as np
from transforms import val_transforms 

def predict(model, img_path, device):

    model.eval()

    model.to(device)

    print(f"Model device: {next(model.parameters()).device}")

    x = Image.open(img_path).convert("RGB")

    input_tensor = val_transforms(x).unsqueeze(0).to(device)

    with torch.no_grad():
        
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = np.round(probs[0, pred].item(),2)

    labels ={0 : "No Diabetic Retinopathy",
             1 : "Mild Diabetic Retinopathy",
              2 : "Moderate Diabetic Retinopathy",
               3 : "Severe Diabetic Retinopathy",
                4 : "Proliferative Diabetic Rethinopathy" }

    print(f"{img_path} → Severity: {labels[pred]} | Confidence : {100*conf:.0f}%")

    return  pred, conf, probs


if __name__ == "__main__":   

    device = "cuda"  

    model = get_student(device=device)      

    for img_id in [2500]:

        img = f"data/images/img_{img_id}.png"
        print(f"\nProcessing: {img}")
        predict(model, img, device)

# %%
