## Diabetic Retinopathy Detection

<p align="center">
  <img width="1594" height="545" alt="image" src="https://github.com/user-attachments/assets/97261dbe-415c-4129-832e-148062f4fd08" />
</p>

*Left: original image.*  
*Right: Grad-CAM visualization highlighting regions influencing the prediction.*


---

# Automated Diabetic Retinopathy Detection

**Diabetic retinopathy (DR)** is an ocular complication of **type 2 diabetes**.  
Chronically high blood sugar damages the tiny blood vessels in the retina, making them leaky: **fats and proteins** escape into the retinal tissue, forming **small yellowish deposits** visible on fundus photographs (*retinography*).  
Without early detection or treatment, this damage can progress to **irreversible vision loss**.

This project aims to Develop a model capable of classifying retinal fundus images according to the **severity level of diabetic retinopathy**, while highlighting **which areas of the retina** influenced the model’s decision.

---

## Web application:
[http://13.60.221.192:8501](http://13.60.221.192:8501)  *Deployed on AWS EC2 using Docker.*


---

## Project Pipeline

The workflow is based on the **[APTOS 2019 Blindness Detection dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection)**.

1. **Data Preprocessing**  
   - Image resizing, normalization, and class balancing.  
   - Data augmentation: rotations, flips, zooms, and random crops.

2. **Model Training**  
   - **Fine-tuning of ResNet-50**, pre-trained on *ImageNet*.  
   - **Knowledge distillation** from ResNet-50 into a lighter **MobileNet** for faster inference.

3. **Model Interpretability**  
   - Generation of **Grad-CAM heatmaps** to visualize which regions of the retina most influenced the model’s output.

4. **Deployment**  
   - Fully containerized with **Docker** for reproducibility.  
   - **Streamlit** web app for testing the model on new fundus images interactively.



## Results and Performance

The model was trained to predict **5 severity classes** of diabetic retinopathy:  
**0 – No DR**, **1 – Mild**, **2 – Moderate**, **3 – Severe**, **4 – Proliferative**.

###  1. Classification Performance

| Model | Accuracy | Quadratic Kappa | Model Size | Inference Time on CPU ( 1000 images) |
|--------|-----------|-----------------|-------------|-------------------------------|
| **ResNet-50 (fine-tuned)** | 0.81 | 0.75 | 94 MB | 7 s |
| MobileNet | 0.74 | 0.72 | 6 MB | 2 s |
| **MobileNet (distilled)** | 0.80 | 0.73 | 6 MB | 2 s |

>  *The Quadratic Weighted Kappa (QWK)* better reflects model agreement across ordinal classes, penalizing large misclassifications more heavily than accuracy.

---

### Deployment

- **Docker**: lightweight, reproducible environment for inference.  
- **Streamlit App**: interactive interface to upload fundus images, view predictions, and Grad-CAM visualizations.  
- **FastAPI Backend**: serves model inference as a REST API endpoint (`/predict`), used by the Streamlit frontend for communication between the UI and the model.

```bash
# Build all services defined in docker-compose.yml
docker compose build

# Run the containers
docker compose up

