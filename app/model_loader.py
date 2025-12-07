import os
import joblib
import logging
from tensorflow.keras.models import load_model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.getcwd(), "models")


class ModelRegistry:
    def __init__(self):
        self.sklearn = {}
        self.vectorizers = {}
        self.keras = {}
        self.transformers = {}

    def load_sklearn(self, key, model_path, vec_path=None):
        model = joblib.load(model_path)
        self.sklearn[key] = model
        if vec_path:
            self.vectorizers[key] = joblib.load(vec_path)

    def load_keras(self, key, model_path, tokenizer_path):
        model = load_model(model_path)
        tokenizer = joblib.load(tokenizer_path)
        self.keras[key] = {"model": model, "tokenizer": tokenizer}

    def load_transformer(self, key, folder):
        tok = AutoTokenizer.from_pretrained(folder)
        model = AutoModelForSequenceClassification.from_pretrained(folder)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        self.transformers[key] = {"tokenizer": tok, "model": model, "device": device}

    def load_all(self):
        # sklearn
        if os.path.exists(f"{MODEL_DIR}/logistic_tfidf.joblib"):
            self.load_sklearn("logistic", f"{MODEL_DIR}/logistic_tfidf.joblib",
                              f"{MODEL_DIR}/tfidf_vectorizer.joblib")

        if os.path.exists(f"{MODEL_DIR}/svm_tfidf.joblib"):
            self.load_sklearn("svm", f"{MODEL_DIR}/svm_tfidf.joblib",
                              f"{MODEL_DIR}/tfidf_vectorizer.joblib")

        if os.path.exists(f"{MODEL_DIR}/xgb_tfidf.joblib"):
            self.load_sklearn("xgb", f"{MODEL_DIR}/xgb_tfidf.joblib",
                              f"{MODEL_DIR}/tfidf_vectorizer.joblib")

        # keras
        if os.path.exists(f"{MODEL_DIR}/bilstm_w2v.h5"):
            self.load_keras("bilstm",
                            f"{MODEL_DIR}/bilstm_w2v.h5",
                            f"{MODEL_DIR}/keras_tokenizer.joblib")

        # transformer
        trans_dir = f"{MODEL_DIR}/transformer/distilbert_finetuned_fast"
        if os.path.exists(trans_dir):
            self.load_transformer("transformer", trans_dir)

    def list_models(self):
        return {
            "sklearn": list(self.sklearn.keys()),
            "keras": list(self.keras.keys()),
            "transformer": list(self.transformers.keys()),
        }


registry = ModelRegistry()
