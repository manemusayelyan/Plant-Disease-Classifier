# Plant Disease FastAPI Service

This repository now includes a deployable FastAPI inference service for the trained plant disease classifier.

## What it serves

- `GET /health`: service health check for Render
- `GET /model-info`: model metadata, classes, image size, normalization
- `POST /predict`: upload an image and get top-k predictions
- `GET /docs`: Swagger UI
- `GET /openapi.json`: OpenAPI schema

## Project structure

- `app/`: FastAPI app and model-loading code
- `checkpoints/plantdiseaseclassifier_efficientnet_b2_bundle.pt`: inference bundle with weights + metadata
- `prepared/label_map.json`: label mapping fallback
- `Dockerfile`: container build for Render
- `render.yaml`: optional Render Blueprint

## Local run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API:

```bash
python -m app.entrypoint
```

4. Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample_leaf.jpg" \
  -F "top_k=5"
```

## Render deployment

### Option A: Use the included `render.yaml`

1. Push this repo to GitHub.
2. In Render, choose `New` -> `Blueprint`.
3. Connect the repo and deploy the Blueprint.
4. After the first deploy finishes, open:

- `https://<your-service>.onrender.com/docs`
- `https://<your-service>.onrender.com/openapi.json`

### Option B: Create the web service manually

1. In Render, choose `New` -> `Web Service`.
2. Connect your GitHub repo.
3. Set:

- Runtime: `Docker`
- Dockerfile path: `./Dockerfile`
- Health check path: `/health`
- Plan: `Free` for demo submission or `Starter` if you want the service to be more reliable and avoid free-tier sleep

4. Deploy.

## Suggested submission URLs

- Production API base URL: `https://<your-service>.onrender.com`
- Swagger/OpenAPI URL: `https://<your-service>.onrender.com/docs`
- Raw schema URL: `https://<your-service>.onrender.com/openapi.json`

## Notes

- The API uses the existing EfficientNet-B2 bundle and reads class names and preprocessing metadata from the checkpoint at startup.
- The Docker build excludes `train/` and `val/` so deployment stays small and fast.
- The model file is about 31.5 MB, so GitHub LFS is optional for this checkpoint size.
- `torch==2.6.0` with `torchvision==0.21.0` is pinned as a compatible pair, using the official CPU wheel index from PyTorch.
