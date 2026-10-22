from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or a .env file."""

    db_name: str = "database/volatility.sqlite"
    model_directory: str = "models"

    class Config:
        env_file = ".env"


settings = Settings()