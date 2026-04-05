# SE-ML Backend

FastAPI backend for Speech-to-Text using local Whisper model + OpenAI GPT post-processing.

## Setup

```bash
cd backend

# สร้าง virtual env
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# ติดตั้ง dependencies
pip install -r requirements.txt

# ตั้งค่า environment
cp .env.example .env
# แก้ไข OPENAI_API_KEY ใน .env
```

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs จะอยู่ที่ → http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/transcribe/` | Transcribe audio (raw Whisper) |
| `POST` | `/api/v1/transcribe/full` | Transcribe + Thai→English cleanup |

### Example (curl)

```bash
# Raw transcription
curl -X POST http://localhost:8000/api/v1/transcribe/ \
  -F "audio=@recording.m4a" \
  -F "language=th"

# Full pipeline (with OpenAI cleanup)
curl -X POST http://localhost:8000/api/v1/transcribe/full \
  -F "audio=@recording.m4a" \
  -F "language=th" \
  -F "initial_prompt=Model, Ensemble, Accuracy, Github"
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app + CORS + routers
│   ├── core/
│   │   └── config.py        # Settings (pydantic-settings, reads .env)
│   ├── api/
│   │   └── routes/
│   │       └── transcribe.py  # /transcribe endpoints
│   └── lib/
│       ├── OPENAI.py          # OpenAI client wrapper
│       └── whisper_handmade.py  # Whisper + GPT pipeline
├── .env.example
├── requirements.txt
└── README.md
```
