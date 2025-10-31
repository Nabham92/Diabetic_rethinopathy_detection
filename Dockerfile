FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p scripts models data

COPY scripts/utils.py scripts/utils.py
COPY scripts/predict.py scripts/predict.py
COPY scripts/main.py scripts/main.py
COPY models/resnet50_aptos19_best.pth models/resnet50_aptos19_best.pth
COPY models/mobilenetv3_distilled_best.pth models/mobilenetv3_distilled_best.pth
COPY data/file_paths_labels.csv data/file_paths_labels.csv
COPY scripts/main.py scripts/app.py scripts/app.py

RUN touch scripts/__init__.py

CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000"]
