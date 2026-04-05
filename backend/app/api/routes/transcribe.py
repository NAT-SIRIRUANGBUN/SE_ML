import os
import tempfile
import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.lib.whisper_handmade import WhisperClient

router = APIRouter(prefix="/transcribe", tags=["transcription"])


class TranscribeResponse(BaseModel):
    text: str


_whisper_client: WhisperClient | None = None


def get_whisper(settings: Settings = Depends(get_settings)) -> WhisperClient:
    global _whisper_client
    if _whisper_client is None:
        _whisper_client = WhisperClient(model=settings.whisper_model)
    return _whisper_client


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, tmp)
    finally:
        tmp.close()
    return tmp.name


# @desc    Upload audio file → Whisper STT → LLM cleanup → return cleaned text
# @route   POST /api/v1/transcribe
# @access  Public
@router.post("/", response_model=TranscribeResponse, summary="Upload audio and get cleaned transcription")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (.wav .mp3 .m4a .ogg .flac .webm)"),
    whisper: WhisperClient = Depends(get_whisper),
):
    tmp_path = _save_upload_to_temp(file)
    try:
        raw_result = whisper.speech_to_text(file_path=tmp_path)
        cleaned_text = whisper.thai_to_english(raw_result)
        return TranscribeResponse(text=cleaned_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
