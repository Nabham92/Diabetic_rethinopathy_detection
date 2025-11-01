import torch
from PIL import Image
import matplotlib.pyplot as plt
from utils import get_student, test_ds,val_transforms
import numpy as np

def predict(model, img_path, device):
    
    """Fait une prédiction sur une image """
    model.eval()
    x = Image.open(img_path).convert("RGB")
    input_tensor = val_transforms(x).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = np.round(probs[0, pred].item(),2)

    print(f"{img_path} → Severity: {pred} | Confidence : {conf:.2f}")

    return pred, conf, probs


if __name__ == "__main__":   
    device = "cuda"  
    weights_path = r"models/mobilenetv3_distilled_best.pth"

    model = get_student(state_dict_path=weights_path, device=device)
    model.eval()       
    model.to(device)
    []
    for img_id in [2500]:
        img = f"data/images/img_{img_id}.png"
        print(f"\nProcessing: {img}")
        predict(model, img, device)

# %%
