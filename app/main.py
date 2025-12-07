import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request
import numpy as np
import torch

# Correct imports for your folder structure
from app.model_loader import registry
from app.utils import clean_text, keras_prepare_sequence
from app.schemas import PredictRequest, BatchPredictRequest, PredictResponse

# -----------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------
app = FastAPI(title="Fake News Detection API", version="1.0")

# -----------------------------------------------------------
# CORS (FULL WORKING VERSION)
# -----------------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=".*",   # ensures wildcard works ALWAYS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# GLOBAL ERROR HANDLER (PREVENTS FAKE CORS ERRORS)
# -----------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": "Server internal error"},
        headers={"Access-Control-Allow-Origin": "*"}  # ensures browser never fakes CORS error
    )

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# -----------------------------------------------------------
# STARTUP LOAD MODELS
# -----------------------------------------------------------
@app.on_event("startup")
def startup_event():
    registry.load_all()
    logger.info("Models loaded: %s", registry.list_models())


@app.get("/health")
def health():
    return {"status": "ok", "models": registry.list_models()}


@app.get("/models")
def model_list():
    return registry.list_models()


# -----------------------------------------------------------
# MODEL HELPERS
# -----------------------------------------------------------
def predict_with_sklearn(model_key: str, text: str):
    model = registry.sklearn.get(model_key)
    vec = registry.vectorizers.get(model_key)

    if model is None or vec is None:
        raise ValueError(f"Missing sklearn model or vectorizer: {model_key}")

    X = vec.transform([text])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[:, 1][0])
    else:
        dec = model.decision_function(X)[0]
        prob = float(1 / (1 + np.exp(-dec)))

    return int(prob > 0.5), prob


def predict_with_keras(model_key: str, text: str):
    info = registry.keras.get(model_key)
    if not info:
        raise ValueError(f"Keras model '{model_key}' missing")

    model = info["model"]
    tokenizer = info["tokenizer"]

    seq = keras_prepare_sequence(tokenizer, [text], max_len=128)
    prob = float(model.predict(seq, verbose=0).ravel()[0])

    return int(prob > 0.5), prob


def predict_with_transformer(model_key: str, text: str):
    info = registry.transformers.get(model_key)
    if not info:
        raise ValueError(f"Transformer model '{model_key}' missing")

    tok = info["tokenizer"]
    model = info["model"]
    device = info["device"]

    inputs = tok(text, truncation=True, padding=True, return_tensors="pt", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    prob = float(probs[1])
    return int(prob > 0.5), prob


# -----------------------------------------------------------
# /PREDICT ENDPOINT
# -----------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty input text")

    cleaned = clean_text(req.text)

    # Choose model
    # TO STABILIZE BACKEND, WE FORCE LOGISTIC MODEL
    model_key = req.model_name or "logistic"

    # fallback if logistic is not loaded
    if model_key not in registry.sklearn:
        available = registry.list_models()
        if available["transformer"]:
            model_key = "transformer"
        elif "xgb" in registry.sklearn:
            model_key = "xgb"
        elif "svm" in registry.sklearn:
            model_key = "svm"
        elif "bilstm" in registry.keras:
            model_key = "bilstm"

    # run prediction
    if model_key in registry.sklearn:
        pred, prob = predict_with_sklearn(model_key, cleaned)

    elif model_key in registry.keras:
        pred, prob = predict_with_keras(model_key, cleaned)

    elif model_key in registry.transformers:
        pred, prob = predict_with_transformer(model_key, cleaned)

    else:
        raise HTTPException(500, f"Model '{model_key}' not available")

    return PredictResponse(prediction=pred, probability=prob, model=model_key)


# -----------------------------------------------------------
# /PREDICT_BATCH ENDPOINT
# -----------------------------------------------------------
@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    output = []

    for t in req.texts:
        try:
            r = predict(PredictRequest(text=t, model_name=req.model_name))
            output.append(r)
        except Exception as e:
            output.append({"error": str(e), "text": t})

    return {"count": len(output), "results": output}
