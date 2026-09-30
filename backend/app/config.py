from pydantic_settings import BaseSettings, SettingsConfigDict
import logging


# class for settings
class Settings(BaseSettings):
    environment: str = "local"
    database_url: str
    gemini_api_key: str
    gemini_model: str = "gemini-3.6-flash"
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"


    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "compliance_copilot"
    llm_provider: str = "gemini"
    groq_generation_model: str = "openai/gpt-oss-120b"



    # write rules telling Settings() how to operate
    model_config = SettingsConfigDict(
        env_file=".env.local",    # look at the files named .env.local
        env_file_encoding="utf-8",      # read it with the UTF-8 test formatting
        extra="ignore",           # ignore any extra variable in .env not listed. don't crash the app
    )


settings = Settings()
if settings.database_url.startswith("postgresql://"):
    settings.database_url = settings.database_url.replace(
        "postgresql://", "postgresql+asyncpg://", 1
    )


# logging
logger = logging.getLogger(__name__)

