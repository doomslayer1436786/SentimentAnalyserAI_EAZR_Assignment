import os
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Configuration & Setup ---
app = FastAPI(
    title="SentimentAnalyzerAI",
    description="API for IMDB Movie Review Sentiment Analysis",
    version="1.0"
)

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Global Variables for Models ---
model_lr = None
vectorizer = None
model_lstm = None
tokenizer = None

# --- Startup Event ---
@app.on_event("startup")
def load_models():
    """Load models into memory on startup."""
    global model_lr, vectorizer, model_lstm, tokenizer
    try:
        model_lr = joblib.load(os.path.join(MODELS_DIR, "model_lr.pkl"))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
        model_lstm = load_model(os.path.join(MODELS_DIR, "model_lstm.h5"))
        tokenizer = joblib.load(os.path.join(MODELS_DIR, "tokenizer.pkl"))
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        # In production, we might want to stop the server here

# --- Preprocessing ---
def preprocess_text(text: str) -> str:
    """Clean and preprocess text similar to training."""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- Pydantic Models ---
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    model_used: str

class MetricsResponse(BaseModel):
    accuracy_lr: float
    accuracy_lstm: float

# --- Endpoints ---

@app.post("/predict", response_model=dict)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment using both Simple (LR) and Advanced (LSTM) models.
    """
    if not model_lr or not model_lstm:
        raise HTTPException(status_code=503, detail="Models not loaded")

    clean_text = preprocess_text(request.text)

    # 1. Logistic Regression Prediction
    vec_text = vectorizer.transform([clean_text])
    pred_lr_class = model_lr.predict(vec_text)[0]
    pred_lr_prob = np.max(model_lr.predict_proba(vec_text))
    
    # 2. LSTM Prediction
    seq_text = pad_sequences(tokenizer.texts_to_sequences([clean_text]), maxlen=100)
    pred_lstm_prob = float(model_lstm.predict(seq_text, verbose=0)[0][0])
    pred_lstm_class = 1 if pred_lstm_prob > 0.5 else 0
    # Adjust confidence for display (0.1 -> 90% neg, 0.9 -> 90% pos)
    lstm_conf = pred_lstm_prob if pred_lstm_class == 1 else (1 - pred_lstm_prob)

    return {
        "original_text": request.text,
        "simple_model": {
            "sentiment": "Positive" if pred_lr_class == 1 else "Negative",
            "confidence": float(round(pred_lr_prob, 4))
        },
        "advanced_model": {
            "sentiment": "Positive" if pred_lstm_class == 1 else "Negative",
            "confidence": float(round(lstm_conf, 4))
        }
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return model performance metrics."""
    # Hardcoded based on your notebook results (replace with real values)
    return {
        "accuracy_lr": 0.86,
        "accuracy_lstm": 0.89
    }