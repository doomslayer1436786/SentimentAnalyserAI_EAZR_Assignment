import os
import joblib
import warnings
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Setup Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Load ML Models (LR only for speed in chatbot) ---
try:
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
    model_lr = joblib.load(os.path.join(MODELS_DIR, "model_lr.pkl"))
    print("✅ ML Models loaded.")
except FileNotFoundError:
    print("❌ Error: Models not found. Run the notebook first.")
    exit()

# --- Load Lightweight LLM (Local) ---
print("⏳ Loading local LLM (distilgpt2)... this may take a moment.")
# Using distilgpt2 as a lightweight free model
pipe = pipeline("text-generation", model="distilgpt2", max_new_tokens=60)
local_llm = HuggingFacePipeline(pipeline=pipe)

# --- Define LangGraph State ---
class BotState(TypedDict):
    user_input: str
    sentiment: Optional[str]
    insights: Optional[str]
    explanation: Optional[str]
    final_response: Optional[str]

# --- Nodes ---

def predict_node(state: BotState):
    """Node 1: Predict Sentiment using the sklearn model."""
    text = state["user_input"]
    vec = vectorizer.transform([text])
    pred = model_lr.predict(vec)[0]
    label = "Positive" if pred == 1 else "Negative"
    return {"sentiment": label}

def insights_node(state: BotState):
    """Node 2: Provide dataset insights based on sentiment."""
    # Hardcoded insights based on typical IMDB analysis
    if state["sentiment"] == "Positive":
        insight = "Common words in positive reviews: 'great', 'fantastic', 'love', 'best'."
    else:
        insight = "Common words in negative reviews: 'bad', 'boring', 'worst', 'waste'."
    return {"insights": insight}

def explain_node(state: BotState):
    """Node 3: Use LLM to explain WHY (Generative)."""
    text = state["user_input"]
    sentiment = state["sentiment"]
    
    template = """
    Review: "{text}"
    Sentiment: {sentiment}
    Task: Explain briefly in one sentence why this review is {sentiment}.
    Explanation:
    """
    prompt = PromptTemplate(template=template, input_variables=["text", "sentiment"])
    
    # Simple chain invocation
    explanation = local_llm(prompt.format(text=text, sentiment=sentiment))
    
    # Clean up artifacts if LLM repeats prompt
    clean_explanation = explanation.split("Explanation:")[-1].strip()
    return {"explanation": clean_explanation}

def response_node(state: BotState):
    """Node 4: Format final output."""
    res = (f"🔍 **Sentiment:** {state['sentiment']}\n"
           f"📊 **Insights:** {state['insights']}\n"
           f"🤖 **AI Explanation:** {state['explanation']}")
    return {"final_response": res}

# --- Build Graph ---
workflow = StateGraph(BotState)

workflow.add_node("predictor", predict_node)
workflow.add_node("insighter", insights_node)
workflow.add_node("explainer", explain_node)
workflow.add_node("responder", response_node)

# Flow: Predict -> Insights -> Explain -> Respond -> End
workflow.set_entry_point("predictor")
workflow.add_edge("predictor", "insighter")
workflow.add_edge("insighter", "explainer")
workflow.add_edge("explainer", "responder")
workflow.add_edge("responder", END)

app = workflow.compile()

# --- Main Interaction Loop ---
if __name__ == "__main__":
    print("\n🎬 Movie Review Sentiment Bot (LangGraph Powered)")
    print("Type a review to analyze (or 'q' to quit).")
    
    while True:
        user_text = input("\nUser: ")
        if user_text.lower() in ['q', 'quit', 'exit']:
            break
            
        inputs = {"user_input": user_text}
        result = app.invoke(inputs)
        print(f"\n{result['final_response']}")