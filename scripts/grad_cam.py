
# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils import val_transforms, get_student
from predict import predict
import cv2


def get_grad_cam_vis(model,img_path):

    target_layer=[model.features[-1]]
    
    pred=[ClassifierOutputTarget(predict(model,img_path,device="cpu")[0])]

    img=Image.open(img_path).convert("RGB")
    img_np=np.array(img)/255
    H,W=img_np.shape[:2]

    input_tensor=val_transforms(img).unsqueeze(0).to("cpu")

    with GradCAM(model,target_layers=target_layer) as cam:
        

        gray_scale_cam=cam(input_tensor,targets=pred)[0,:]
        gray_scale_cam=cv2.resize(gray_scale_cam,(W,H))
        print(gray_scale_cam.min(), gray_scale_cam.max())
        visu=show_cam_on_image(img_np,gray_scale_cam,use_rgb=True)

    return(visu)

def plot_grad_cam(model,img_path):

    visu=get_grad_cam_vis(model,img_path)
    img=Image.open(img_path).convert("RGB")

    fig,ax=plt.subplots(1,2,figsize=(16,12))
    ax[0].imshow(visu)
    ax[0].axis("off")
    ax[1].imshow(img)
    plt.axis("off")
    plt.show()

if __name__=="__main__":
    img_path = r"data/images/img_3623.png"
    weights_path = r"models/mobilenetv3_distilled_best.pth"
    model = get_student(weights_path, device="cpu")
    plot_grad_cam(model,img_path)


# %%
