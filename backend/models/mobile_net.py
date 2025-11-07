from torchvision.models import mobilenet_v3_small
import torch.nn as nn 
import torch 

best_model_path=r"backend/models/mobilenetv3_distilled_best.pth"

def get_student(state_dict_path=best_model_path,device="cpu"):

    mobile_net=mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1")

    mobile_net.classifier[3]=nn.Linear(1024,5)     
    
    if state_dict_path : 

        state_dict = torch.load(state_dict_path, map_location=device)
        mobile_net.load_state_dict(state_dict)

    return(mobile_net)

