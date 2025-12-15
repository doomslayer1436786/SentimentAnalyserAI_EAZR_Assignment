from flask import Flask, request, render_template_string
import requests

app = Flask(__name__)

# FastAPI backend endpoint
BACKEND_URL = "http://localhost:8090/predict"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sentiment Analyzer AI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f6f8;
        }
        .container {
            width: 600px;
            margin: 40px auto;
            background: #ffffff;
            padding: 25px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
        }
        textarea {
            width: 100%;
            padding: 10px;
            font-size: 14px;
            resize: vertical;
        }
        button {
            margin-top: 12px;
            width: 100%;
            padding: 10px;
            font-size: 15px;
            cursor: pointer;
        }
        .results {
            margin-top: 20px;
        }
        .boxed {
            background: #f1f1f1;
            padding: 10px;
            border-radius: 4px;
        }
        .error {
            margin-top: 10px;
            color: red;
            font-weight: bold;
        }
        .model {
            margin-top: 15px;
            padding: 10px;
            border-left: 4px solid #007BFF;
            background: #f9f9f9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>IMDB Sentiment Analyzer</h1>

        <form method="post">
            <textarea name="text" rows="6" placeholder="Enter a movie review...">{{ text or "" }}</textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        {% if result %}
            <div class="results">
                <h2>Results</h2>

                <p><strong>Original Text:</strong></p>
                <p class="boxed">{{ result.original_text }}</p>

                <div class="model">
                    <h3>Simple Model (Logistic Regression)</h3>
                    <p>Sentiment: <strong>{{ result.simple_model.sentiment }}</strong></p>
                    <p>Confidence: {{ result.simple_model.confidence }}</p>
                </div>

                <div class="model">
                    <h3>Advanced Model (LSTM)</h3>
                    <p>Sentiment: <strong>{{ result.advanced_model.sentiment }}</strong></p>
                    <p>Confidence: {{ result.advanced_model.confidence }}</p>
                </div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if not text:
            error = "Please enter some text."
        else:
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"text": text},
                    timeout=5
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                error = f"Backend error: {e}"

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        error=error,
        text=text
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
