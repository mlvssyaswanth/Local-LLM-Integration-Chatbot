# FastAPI app - health + chat endpoint

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from api.schemas import ChatRequest, ChatResponse, HealthResponse
from model.inference import RecipeInferenceEngine

logger = logging.getLogger(__name__)

engine: RecipeInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = RecipeInferenceEngine(model_name=config.OLLAMA_MODEL)
    logger.info("Recipe inference engine ready (model=%s)", config.OLLAMA_MODEL)
    yield
    engine = None


app = FastAPI(
    title="Recipe Chatbot API",
    description="Local LLM recipe suggestion API. Accepts ingredients and returns recipe suggestions based on the dataset.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model=config.OLLAMA_MODEL)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready")
    try:
        response_text = engine.suggest_recipe(request.message)
        return ChatResponse(response=response_text)
    except RuntimeError as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=503, detail=str(e)) from e
