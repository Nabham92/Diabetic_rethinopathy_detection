from fastapi import FastAPI, File, UploadFile
import tempfile
from PIL import Image
import io
from .predict import predict
from .utils import get_student

app = FastAPI()
weights_path = "models/mobilenetv3_distilled_best.pth"

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Sauvegarder temporairement le fichier pour obtenir un chemin
    contents = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Charger le modèle
    model = get_student(weights_path, device="cpu")

    # Appeler ta fonction avec le chemin temporaire
    severity, conf = predict(model, tmp_path, device="cpu")

    return {"Image": tmp_path, "Severity": severity, "Confidence": conf}
