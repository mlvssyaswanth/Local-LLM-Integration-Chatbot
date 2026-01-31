# env config - override with env vars if needed

import os

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8000"))

CHATBOT_HOST = os.environ.get("CHATBOT_HOST", "127.0.0.1")
CHATBOT_PORT = int(os.environ.get("CHATBOT_PORT", "5000"))

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
