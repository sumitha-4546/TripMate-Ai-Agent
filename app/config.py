import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "destinations"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env without overriding pre-set test vars
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


class Settings:

    def __init__(self) -> None:
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.weather_api_base_url: str = os.getenv(
            "WEATHER_API_BASE_URL", "https://api.open-meteo.com"
        ).rstrip("/")
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.app_env: str = os.getenv("APP_ENV", "development")
        self.app_port: int = int(os.getenv("APP_PORT", "8000"))
        self.app_host: str = os.getenv("APP_HOST", "0.0.0.0")
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.base_dir: Path = BASE_DIR
        self.data_dir: Path = DATA_DIR
        self.vector_store_dir: Path = VECTOR_STORE_DIR
        self.logs_dir: Path = LOGS_DIR

    @property
    def is_api_key_configured(self) -> bool:
        key = self.gemini_api_key.strip()
        return bool(key) and "your_gemini_api_key" not in key

    def safe_dict(self) -> dict:
        return {
            "gemini_model": self.gemini_model,
            "has_gemini_key": self.is_api_key_configured,
            "weather_api_base_url": self.weather_api_base_url,
            "embedding_model": self.embedding_model,
            "app_env": self.app_env,
            "data_dir": str(self.data_dir),
            "vector_store_dir": str(self.vector_store_dir),
        }


settings = Settings()
