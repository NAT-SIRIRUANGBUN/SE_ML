import os
import tempfile
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.lib.whisper_handmade import WhisperClient

router = APIRouter(prefix="/transcribe", tags=["transcription"])

_whisper_client: WhisperClient | None = None

def get_whisper(settings: Settings = Depends(get_settings)) -> WhisperClient:
    global _whisper_client
    if _whisper_client is None:
        _whisper_client = WhisperClient(model=settings.whisper_model)
    return _whisper_client


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}
jobs: dict = {}


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


def process_audio(job_id: str, file_path: str, whisper: WhisperClient):
    try:
        jobs[job_id]["status"] = "processing"
        raw_result = whisper.speech_to_text(file_path=file_path)
        cleaned_text = whisper.thai_to_english(raw_result)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = cleaned_text
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


@router.post("/")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (.wav .mp3 .m4a .ogg .flac .webm)"),
    whisper: WhisperClient = Depends(get_whisper),
):
    tmp_path = _save_upload_to_temp(file)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}
    
    background_tasks.add_task(process_audio, job_id, tmp_path, whisper)
    return {"job_id": job_id, "status": "queued"}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
