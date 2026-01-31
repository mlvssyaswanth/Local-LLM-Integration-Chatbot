# start the web chatbot (serves the page, page hits main API)

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "chatbot.web.app:app",
        host=config.CHATBOT_HOST,
        port=config.CHATBOT_PORT,
        reload=False,
    )
