from huggingface_hub import InferenceClient
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN") 
print(HF_TOKEN) # fix your env var name spelling if needed

client = InferenceClient( provider="novita",token=HF_TOKEN)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    
    messages = [{"role": "user", "content": user_text}]
    
    completion = client.chat.completions.create(
  model="mistralai/Mistral-7B-Instruct-v0.3",
  messages=messages
)

    response_text = completion.choices[0].message["content"]
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
