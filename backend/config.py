from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = ""
    gemini_api_key: str = ""
    mistral_api_key: str = ""
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model_name: str = "meta-llama/Llama-3-8B-Instruct"
    federated_secret_key: str = "change_this_secret_key"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
