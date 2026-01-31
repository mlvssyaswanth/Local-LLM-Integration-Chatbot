#!/usr/bin/env bash
# set -e
# cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2:1b
python run_api.py
