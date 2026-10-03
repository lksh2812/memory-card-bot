"""Configuration, read once from environment variables (and a .env file)."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(override=True)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/memory_game"
        )
    )
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    deepgram_api_key: str = field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))
    deepgram_voice: str = field(default_factory=lambda: os.getenv("DEEPGRAM_VOICE", "aura-2-helena-en"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    # Groq retired the Llama 3.x models in Aug 2026; gpt-oss-20b is the fast replacement.
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))

    start_length: int = field(default_factory=lambda: _int("GAME_START_LENGTH", 3))
    max_length: int = field(default_factory=lambda: _int("GAME_MAX_LENGTH", 10))
    lives: int = field(default_factory=lambda: _int("GAME_LIVES", 3))


settings = Settings()
