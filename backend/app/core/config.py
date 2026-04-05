from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # --- OpenAI ---
    openai_api_key: str

    # --- Whisper ---
    whisper_model: str = "large"          # tiny | base | small | medium | large
    whisper_language: str = "th"
    whisper_initial_prompt: str | None = None

    # --- App ---
    app_name: str = "SE-ML STT Backend"
    debug: bool = False
    allowed_origins: list[str] = ["*"]     # ล็อคให้แคบลงตอน production

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once)."""
    return Settings()
