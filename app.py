import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_API_KEY")
if not HF_TOKEN:
    raise ValueError("HF_API_KEY not set in .env")

# Use the correct router endpoint for chat completions
HF_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided."}), 400

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 200
    }

    response = requests.post(HF_URL, headers=headers, json=payload)
    print("Status:", response.status_code)


    try:
        result = response.json()
    except ValueError:
        return jsonify({"error": "Invalid response from HF", "raw": response.text}), 500

    if "error" in result:
        return jsonify(result), 500

    reply = result["choices"][0]["message"]["content"]
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
