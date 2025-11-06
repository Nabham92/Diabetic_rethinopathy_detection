import streamlit as st
import requests
from PIL import Image
from backend.grad_cam import get_grad_cam_vis
from scripts.utils import get_student
import io
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Détection de rétinopathie", page_icon="🩺")

st.title("🩺 Diabetic Retinopathy Detection")
#st.write("")

BASE_API_URL = os.getenv("API_URL","http://localhost:8000")


weights_path = r"models/mobilenetv3_distilled_best.pth"
model = get_student(weights_path, device="cpu")

uploaded_file = st.file_uploader("Drag an image :", type=["png", "jpg", "jpeg"])

labels ={0 : "No Diabetic Retinopathy",
             1 : "Mild Diabetic Retinopathy",
              2 : "Moderate Diabetic Retinopathy",
               3 : "Severe Diabetic Retinopathy",
                4 : "Proliferative Diabetic Rethinopathy" }

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Image originale", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name  # vrai chemin disque

    # Request the API 
    with st.spinner("Analyse en cours..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        predict_url=BASE_API_URL+"/predict"
        response = requests.post(predict_url, files=files)

        st.write("Code retour API :", response.status_code)
        st.write("Texte brut :", response.text)

    if response.status_code == 200:
        result = response.json()
        st.success("✅ Analyse terminée !")

        st.write(f"**Sévérité prédite :** {labels[result['Severity']]}")
        st.write(f"**Confiance :** {result['Confidence']:.2f}")

        grad_cam_vis=get_grad_cam_vis(model,tmp_path)
        with col2:
            st.image(grad_cam_vis, caption="Grad-CAM", use_container_width=True)
        
    else:
        st.error(f"Erreur {response.status_code} : impossible d’obtenir une réponse.")
else:
    st.info("⬆️ Glisse une image pour commencer.")
