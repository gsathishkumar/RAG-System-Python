import os

from pydantic import BaseModel, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ENV = os.getenv("APP_ENV", "dev")

class DatabaseSettings(BaseModel):
    host: str
    port: int
    user: str
    password: str
    name: str

class FileUploadSettings(BaseModel):
    allowed_file_types: list[str] = ["application/pdf"]
    max_file_size: int = 1024
    upload_path: str = "uploads/pdfs"

    @computed_field(return_type=int)
    @property
    def max_file_size_in_kb(self):
        return self.max_file_size * 1024
    
class Settings(BaseSettings):
    app_name: str
    app_port: int
    gemini_api_key: str | None = None
    openai_api_key: str | None = None

    db: DatabaseSettings
    file: FileUploadSettings

    model_config = SettingsConfigDict(
        env_file=f".env.{APP_ENV}",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore" # Ignore extra fields in the .env file
    )

settings = Settings()