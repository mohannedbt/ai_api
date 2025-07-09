
from flask import Flask, request, jsonify
from huggingface_hub import HuggingFaceOpenAI

import os
app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKN")


client = HuggingFaceOpenAI(token=HF_TOKEN)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    messages = [{"role": "user", "content": user_text}]

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages
    )
    response_text = completion.choices[0].message["content"]
    return jsonify({"response": response_text})
