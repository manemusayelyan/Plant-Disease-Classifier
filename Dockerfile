FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY checkpoints/plantdiseaseclassifier_efficientnet_b2_bundle.pt ./checkpoints/plantdiseaseclassifier_efficientnet_b2_bundle.pt
COPY prepared/label_map.json ./prepared/label_map.json

EXPOSE 10000

CMD ["python", "-m", "app.entrypoint"]
