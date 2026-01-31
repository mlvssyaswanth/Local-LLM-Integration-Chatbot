# serves the web chat page - page then calls the main API

import argparse
import sys
from pathlib import Path

import uvicorn

import config

# so we can run from project root or elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

app = FastAPI(
    title="Recipe Chatbot",
    description="Web UI for the recipe chatbot.",
    docs_url=None,
    redoc_url=None,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if not INDEX_HTML.exists():
        return HTMLResponse(
            "<h1>Not found</h1><p>index.html missing from chatbot/web/static/</p>",
            status_code=404,
        )
    return FileResponse(INDEX_HTML, media_type="text/html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start recipe chatbot web UI")
    parser.add_argument("--host", default=config.CHATBOT_HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=config.CHATBOT_PORT, help="Port")
    args = parser.parse_args()
    uvicorn.run(
        "chatbot.web.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
