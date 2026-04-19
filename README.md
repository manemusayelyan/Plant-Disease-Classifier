# Plant Disease Classifier
This repository provides a production-ready FastAPI inference service for the Plant Disease Classifier, utilizing a fine-tuned EfficientNet-B2 architecture.

## Live Demo & API Access
The service is deployed on Render and is publicly accessible.

Interactive Swagger UI: https://plant-disease-classifier-lnh2.onrender.com/docs

Health Check: https://plant-disease-classifier-lnh2.onrender.com/health

Model Info: https://plant-disease-classifier-lnh2.onrender.com/model-info

## Features 

Inference Pipeline: Automated image resizing (260x260) and normalization based on training metadata.

EfficientNet-B2 Backbone: Optimized for a balance between accuracy and inference speed.

Dockerized: Containerized using a slim Python image to ensure reproducibility across environments.

Validation Fallback: Integrated label mapping fallback to ensure robust class identification.

