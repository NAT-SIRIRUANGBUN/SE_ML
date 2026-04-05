from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.api.routes import transcribe

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Whisper model immediately on startup
    print(f"🔄 Loading Whisper model '{settings.whisper_model}' ...")
    transcribe._whisper_client = transcribe.WhisperClient(model=settings.whisper_model)
    print("✅ Whisper model loaded!")
    yield
    transcribe._whisper_client = None

app = FastAPI(
    title=settings.app_name,
    description="Speech-to-Text backend powered by OpenAI Whisper + GPT post-processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe.router, prefix="/api/v1")


# @desc    Health check
# @route   GET /health
# @access  Public
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "app": settings.app_name}
