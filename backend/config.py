from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key:       str = ""
    mistral_api_key:    str = ""
    deepseek_api_key:   str = ""
    openrouter_api_key: str = ""
    vllm_base_url:      str = "http://localhost:8001/v1"
    vllm_model_name:    str = "meta-llama/Llama-3-8B-Instruct"
    federated_secret_key: str = "change_this_secret_key"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
