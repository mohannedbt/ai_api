# AI API — LLM Chat Gateway

Lightweight Flask API exposing a `/chat` endpoint to interact with Large Language Models through the Hugging Face Router.
The service acts as a backend abstraction layer between applications and hosted LLM providers.

---

## Overview

This API provides:

* Simple REST interface for AI chat
* Secure environment-based API key handling
* Model abstraction layer
* Easy integration into web or backend systems

Architecture:

```
Client → ai_api → Hugging Face Router → LLM → Response
```

---

## Tech Stack

* Python
* Flask
* Requests
* python-dotenv
* Hugging Face Inference Router
* Llama 3.2 Instruct model

---

## Installation

### Clone repository

```bash
git clone https://github.com/yourusername/ai_api.git
cd ai_api
```

### Create virtual environment

```bash
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install flask requests python-dotenv
```

---

## Environment Configuration

Create a `.env` file:

```
HF_API_KEY=your_huggingface_token
```

---

## Running the Server

```bash
python app.py
```

Server runs at:

```
http://localhost:8000
```

---

## API Usage

### POST `/chat`

Request:

```json
{
  "message": "Explain transformers in simple terms"
}
```

Response:

```json
{
  "reply": "Model response text"
}
```

---

## Example Client

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Hello"}
)

print(response.json())
```

---

## Model Configuration

Current model:

```
meta-llama/Llama-3.2-1B-Instruct
```

Change model in the source code:

```python
MODEL = "your-model-name"
```

---

## Security Notes

* API key stored in environment variables
* Client applications never access provider tokens directly
* Backend controls LLM access

---

## Possible Extensions

* Streaming responses
* Conversation memory
* Authentication layer
* Rate limiting
* Multi-model routing
* Logging and monitoring

---

## Author

Mohanned Bentaleb
Software Engineering — AI Systems and Backend Development
