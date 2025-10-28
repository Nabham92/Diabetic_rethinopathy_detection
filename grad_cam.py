
# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils import test_ds,df_test

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def select_indices_by_class(df, targets, n_per_class):
    indices = []
    for target in targets:
        class_indices = np.where(np.array(df['label']) == target)[0]
        if len(class_indices) > 0:
            chosen = np.random.choice(class_indices, size=min(n_per_class, len(class_indices)), replace=False)
            indices.extend(chosen)
    return indices

def compute_gradcam_for_model(model, target_layers, input_tensor, label, rgb_img, device):
    model.to(device)
    model.eval()

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets_cam = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets_cam)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        y_pred = torch.argmax(model(input_tensor), dim=1).item()

    return visualization, y_pred


def plot_gradcam_results(images_data, models_info):
    n_rows = len(images_data)
    n_cols = len(models_info) + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (rgb_img, label, visualizations) in enumerate(images_data):
        # Image originale
        axes[row, 0].imshow(rgb_img)
        axes[row, 0].set_title(f"Image (y={label})", fontsize=10)
        axes[row, 0].axis("off")

        # Visualisations GradCAM
        for col, (name, (vis, y_pred)) in enumerate(visualizations.items(), start=1):
            axes[row, col].imshow(vis)
            axes[row, col].set_title(f"{name}\n(pred={y_pred})", fontsize=10)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def plot_gradcam_multi_model(
    models,
    model_names,
    df,
    dataset,
    target_layers_list,
    targets=[0, 1, 2],
    n_per_class=2,
    img_size=256,
    device="cuda"
):

    assert len(models) == len(model_names) == len(target_layers_list), \
        "models, model_names et target_layers_list doivent avoir la même longueur."

    # 1️⃣ Sélection des indices à afficher
    indices = select_indices_by_class(df, targets, n_per_class)
    if len(indices) == 0:
        print("Aucune image trouvée pour ces classes :", targets)
        return

    images_data = []

    # 2️⃣ Pour chaque image sélectionnée, calculer les activations pour tous les modèles
    for idx in indices:
        input_tensor, label = dataset[idx]
        input_tensor = input_tensor.unsqueeze(0).to(device)
        rgb_img = np.array(Image.open(df["img_path"].iloc[idx]).resize((img_size, img_size))) / 255.0

        visualizations = {}
        for model, name, target_layers in zip(models, model_names, target_layers_list):
            vis, pred = compute_gradcam_for_model(model, target_layers, input_tensor, label, rgb_img, device)
            visualizations[name] = (vis, pred)

        images_data.append((rgb_img, label, visualizations))

    # 3️⃣ Affichage
    plot_gradcam_results(images_data, list(zip(model_names, models)))

    plt.show()

# %%
if __name__=="__main__":

    from utils import get_student,get_teacher

    student_path=r"mobilenetv3_distilled_best.pth"
    teacher_path=r"resnet50_aptos19_best.pth"
    student=get_student(student_path).to("cuda")
    teacher=get_teacher(teacher_path)

    models = [teacher, student]
    model_names = ["Teacher","Student"]
    target_layers_list = [
        [teacher.layer4[-1]],
        [student.features[-1]]
    ]

    plot_gradcam_multi_model(
        models=models,
        model_names=model_names,
        df=df_test,
        dataset=test_ds,
        target_layers_list=target_layers_list,
        targets=[4],
        n_per_class=1,
        img_size=256
    )
# %%
