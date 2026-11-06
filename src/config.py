from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class _BaseCfg(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class OpenSearchSettings(_BaseCfg):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "ai-docs"
    chunk_index_suffix: str = "chunks"
    vector_dimension: int = 1024
    vector_space_type: str = "cosinesimil"
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2


class OllamaSettings(_BaseCfg):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OLLAMA__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:11434"
    model: str = "qwen3:4b"
    timeout: int = 300


class EmbeddingSettings(_BaseCfg):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="EMBEDDING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    provider: Literal["bailian", "ollama"] = "bailian"

    bailian_api_key: str = ""
    bailian_model: str = "text-embedding-v3"
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    ollama_model: str = "qwen3-embedding"

    dimensions: int = 1024


class LangfuseSettings(_BaseCfg):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="LANGFUSE__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:4001"
    enabled: bool = False
    flush_at: int = 15
    flush_interval: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    debug: bool = False


class RedisSettings(_BaseCfg):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="REDIS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    decode_responses: bool = True
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    ttl_hours: int = 6


class ChunkingSettings(_BaseCfg):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    chunk_size: int = 600
    overlap_size: int = 100
    min_chunk_size: int = 100
    section_based: bool = True


class DocumentSettings(_BaseCfg):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="DOCUMENTS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    upload_dir: str = "uploads"
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf", ".md", ".markdown", ".txt", ".rst"]


class Settings(_BaseCfg):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "ai-docs-rag"

    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    documents: DocumentSettings = Field(default_factory=DocumentSettings)


def get_settings() -> Settings:
    return Settings()
