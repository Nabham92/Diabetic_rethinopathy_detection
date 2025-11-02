FROM python:3.12

WORKDIR /app

# Dépendances pour open-cv
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit 

RUN mkdir -p scripts models data

COPY data/file_paths_labels.csv data/file_paths_labels.csv


COPY scripts/grad_cam.py scripts/grad_cam.py
COPY scripts/app.py scripts/app.py
COPY scripts/predict.py scripts/predict.py
COPY scripts/utils.py scripts/utils.py

COPY models/mobilenetv3_distilled_best.pth models/mobilenetv3_distilled_best.pth

RUN touch scripts/__init__.py

EXPOSE 8501

CMD ["streamlit", "run", "scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
