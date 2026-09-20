"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Research Paper Intelligence Platform"
    data_dir: Path = Path("data")
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "mock"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    retrieval_top_k: int = 5
    retrieval_threshold: float = 0.25
    max_upload_mb: int = 25

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def database_path(self) -> Path:
        return self.data_dir / "research.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def vectorstore_dir(self) -> Path:
        return self.data_dir / "vectorstore"

    def ensure_directories(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
