from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "mistral"
    api_key: str = ""
    log_level: str = "INFO"
    relevance_threshold: int = 2
    arxiv_max_results: int = 5


settings = Settings()
