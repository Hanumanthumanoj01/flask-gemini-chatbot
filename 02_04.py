from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

app = Flask(__name__)

# -------------------------
# Gemini AI Setup
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY not found!")

genai.configure(api_key=GEMINI_API_KEY)

# -------------------------
# Model
# -------------------------
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# -------------------------
# Prompt (UPDATED: Unlocked the AI's Brain)
# -------------------------
GLOBAL_PROMPT_TEMPLATE = """
You are Mr. Manoj, a highly intelligent, natural, and helpful AI assistant.

Guidelines:
- Provide clear, natural, and highly accurate answers.
- Use Markdown formatting (bolding, bullet points, headers, code blocks) naturally to make your responses easy to read.
- If the user asks a conversational question, be friendly and concise.
- If the user asks a technical or complex question (like Engineering, Indian Railways, Science), provide a deep, well-structured, and highly informative explanation.
- Adapt your response structure to the user's prompt (do not force 'Conclusion' or 'Features' headers unless the prompt specifically warrants it).

Question:
{question}

Answer:
"""

# -------------------------
# Rate Limiting
# -------------------------
LAST_REQUEST_TIME = {}

def is_allowed(ip, cooldown=2):
    now = time.time()
    last = LAST_REQUEST_TIME.get(ip, 0)

    if now - last < cooldown:
        return False

    LAST_REQUEST_TIME[ip] = now
    return True

# -------------------------
# Gemini Query
# -------------------------
def query_llm(question):
    prompt_text = GLOBAL_PROMPT_TEMPLATE.format(question=question)

    try:
        response = model.generate_content(
            prompt_text,
            generation_config={
                "temperature": 0.6,  # Increased from 0.3 for more intelligent/natural reasoning
                "top_p": 0.9,
                "max_output_tokens": 2048
            },
            stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except ResourceExhausted:
        yield "⚠️ API quota exceeded. Please try again later."
    except Exception as e:
        print("ERROR:", e)
        yield f"⚠️ Error: {str(e)}"

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    ip = request.remote_addr

    if not is_allowed(ip):
        return "⏳ Please wait 2 seconds before sending another message.", 429

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return "Please ask me something!", 400

    return app.response_class(query_llm(question), mimetype='text/plain')

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)