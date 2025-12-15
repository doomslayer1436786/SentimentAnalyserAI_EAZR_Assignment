# SentimentAnalyzer AI-Assessment

## Overview
An end-to-end Machine Learning application for sentiment analysis on IMDB movie reviews. This project includes a training pipeline, a FastAPI backend, and a LangGraph-powered conversational interface.

## Project Structure
- `notebooks/`: Jupyter notebook for EDA, Training, and Evaluation.
- `src/models/`: Serialized models (Logistic Regression, LSTM, Vectorizers).
- `src/api/`: FastAPI application.
- `src/chatbot/`: LangGraph chatbot script.

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Virtual Environment recommended

### 2. Installation 
open command line
# Create virtual env
python -m venv venv
source venv/Scripts/activate  # or venv\bin\activate on Windows

# Install dependencies
pip install -r requirements.txt

### 3. Running the App

**Step 1: Train Models**
Run the notebook in `notebooks/notebook.ipynb` to generate `.pkl` and `.h5` files. Ensure these files are saved in `src/models/`.

**Step 2: Start API**

uvicorn src.api.app:app --reload --port 8090

**Step 3: Start Frontend in a new terminal** 

python Frontend\app.py

