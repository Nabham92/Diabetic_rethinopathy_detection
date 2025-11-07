from fastapi import FastAPI, File, UploadFile
import tempfile
from PIL import Image
import io, base64
import numpy as np
from predict import predict
from models.mobile_net import get_student
from grad_cam import get_grad_cam_vis

app = FastAPI()

weights_path = "models/mobilenetv3_distilled_best.pth"
device = "cpu"
model = get_student(weights_path, device=device)

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    severity, conf, _ = predict(model, tmp_path, device=device)
    grad_cam_img = get_grad_cam_vis(model, tmp_path)

    # ✅ Convertir NumPy → PIL si besoin
    if isinstance(grad_cam_img, np.ndarray):
        grad_cam_img = Image.fromarray(grad_cam_img)

    # Encoder l’image en Base64
    buffered = io.BytesIO()
    grad_cam_img.save(buffered, format="PNG")
    grad_cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "Severity": severity,
        "Confidence": conf,
        "GradCAM": grad_cam_base64
    }
