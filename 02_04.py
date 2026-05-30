from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import time
import base64
import io
from PIL import Image
from dotenv import load_dotenv

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
# AUTOMATIC MODEL DETECTION (THE ULTIMATE FIX)
# We no longer guess names. We ask Google exactly what models your key owns!
# -------------------------
FALLBACK_MODELS = []
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            # We put the fast 'flash' models at the front of the list
            if 'flash' in m.name.lower():
                FALLBACK_MODELS.insert(0, m.name)
            else:
                FALLBACK_MODELS.append(m.name)
    print(f"✅ Automatically detected allowed models: {FALLBACK_MODELS}")
except Exception as e:
    print("⚠️ Failed to fetch models automatically. Using default.", e)
    FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

# -------------------------
# Prompts
# -------------------------
GLOBAL_PROMPT_TEMPLATE = """
You are Mr. Manoj, a highly intelligent, natural, and helpful AI assistant.

Guidelines:
- Provide clear, natural, and highly accurate answers.
- Use Markdown formatting (bolding, bullet points, headers, code blocks) naturally.
- If the user provides an image, analyze it deeply and answer their questions about it.
- If the user asks a technical question, provide a well-structured explanation.

Question:
{question}

Answer:
"""

LAST_REQUEST_TIME = {}


def is_allowed(ip, cooldown=2):
    now = time.time()
    last = LAST_REQUEST_TIME.get(ip, 0)
    if now - last < cooldown:
        return False
    LAST_REQUEST_TIME[ip] = now
    return True


def query_llm(question, image_b64=None):
    prompt_text = GLOBAL_PROMPT_TEMPLATE.format(question=question)
    payload = [prompt_text]

    if image_b64:
        try:
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes))
            payload.append(img)
        except Exception as e:
            print(f"Error processing image: {e}")

    for model_name in FALLBACK_MODELS:
        try:
            print(f"Attempting chat with {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                payload,
                generation_config={"temperature": 0.6, "top_p": 0.9, "max_output_tokens": 2048},
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return

        except Exception as e:
            print(f"⚠️ {model_name} failed: {str(e)}")
            continue

            # If all allowed models are out of quota
    yield "⚠️ All AI servers are currently busy or out of quota. Please wait 1 minute and try again."


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
    image_b64 = data.get("image", None)

    if not question and not image_b64:
        return "Please ask me something or upload an image!", 400

    return app.response_class(query_llm(question, image_b64), mimetype='text/plain')


# Quick Translation Route for Voice Feature
@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text")
    target_lang = data.get("language")

    if not text or not target_lang:
        return jsonify({"error": "Missing data"}), 400

    clean_text = text.replace('*', '').replace('#', '')
    translation_prompt = f"Translate the following text into {target_lang}. Respond ONLY with the translated text, do not add any conversational filler. Text to translate: {clean_text}"

    for model_name in FALLBACK_MODELS:
        try:
            print(f"Attempting translation with {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(translation_prompt)
            return jsonify({"translated_text": response.text})
        except Exception as e:
            print(f"⚠️ Translation {model_name} failed: {str(e)}")
            continue

    return jsonify({"error": "Translation servers busy"}), 429


if __name__ == "__main__":
    app.run(debug=True)