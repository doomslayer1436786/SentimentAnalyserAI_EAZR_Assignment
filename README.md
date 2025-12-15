# SentimentAnalyzer AI-Assessment

## Overview
An end-to-end Machine Learning application for sentiment analysis on IMDB movie reviews. This project includes a training pipeline, a FastAPI backend, and a LangGraph-powered conversational interface.

## Project Structure
- `data/`: Dataset storage.
- `notebooks/`: Jupyter notebook for EDA, Training, and Evaluation.
- `src/models/`: Serialized models (Logistic Regression, LSTM, Vectorizers).
- `src/api/`: FastAPI application.
- `src/chatbot/`: LangGraph chatbot script.

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Virtual Environment recommended

### 2. Installation
```bash
# Create virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

