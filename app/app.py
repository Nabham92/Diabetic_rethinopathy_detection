import streamlit as st
import requests
from PIL import Image
import io
import tempfile
import os
from dotenv import load_dotenv
import base64

load_dotenv()

st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="")

st.title("Diabetic Retinopathy Detection")

#st.write("")

BASE_API_URL = os.getenv("API_URL","http://localhost:8000")

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
        st.image(image, caption="Original Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Request the API 
    with st.spinner("Analyzing"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        predict_url=BASE_API_URL+"/predict"
        response = requests.post(predict_url, files=files)

        st.write("API response :", response.status_code)

    if response.status_code == 200:
        result = response.json()
        st.success(" Analysis completed")

        st.write(f"**Predicted severity :** {labels[result['Severity']]}")
        st.write(f"**Confidence :** {result['Confidence']:.2f}")

        #grad_cam_vis=get_grad_cam_vis(model,tmp_path)
        grad_cam_bytes = base64.b64decode(result["GradCAM"])
        grad_cam_img = Image.open(io.BytesIO(grad_cam_bytes))

        with col2:
            st.image(grad_cam_img, caption="Grad-CAM", use_container_width=True)
        
    else:
        st.error(f"Error {response.status_code}")
else:
    st.info("Drag an image")
