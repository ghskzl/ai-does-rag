import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from src.config import get_settings
from src.routers import agentic_ask, documents, ping, search
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan for the API."""
    logger.info("Starting AI Docs RAG API...")

    settings = get_settings()
    app.state.settings = settings

    try:
        opensearch_client = make_opensearch_client()
        app.state.opensearch_client = opensearch_client

        if opensearch_client.health_check():
            logger.info("OpenSearch connected successfully")

            setup_results = opensearch_client.setup_indices(force=False)
            if setup_results.get("chunks_index"):
                logger.info("Chunks index created")
            else:
                logger.info("Chunks index already exists")

            try:
                stats = opensearch_client.client.count(index=opensearch_client.index_name)
                logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
            except Exception:
                logger.info("OpenSearch index ready (stats unavailable)")
        else:
            logger.warning("OpenSearch connection failed - search features will be limited")
    except Exception as e:
        logger.error(f"OpenSearch initialization failed: {e}")
        app.state.opensearch_client = None

    try:
        app.state.embeddings_service = make_embeddings_service()
        logger.info("Embeddings service initialized")
    except Exception as e:
        logger.error(f"Embeddings service initialization failed: {e}")
        app.state.embeddings_service = None

    try:
        app.state.ollama_client = make_ollama_client()
        logger.info("Ollama client initialized")
    except Exception as e:
        logger.error(f"Ollama client initialization failed: {e}")
        app.state.ollama_client = None

    try:
        app.state.langfuse_tracer = make_langfuse_tracer()
        logger.info("Langfuse tracer initialized")
    except Exception as e:
        logger.error(f"Langfuse tracer initialization failed: {e}")
        app.state.langfuse_tracer = None

    try:
        app.state.cache_client = make_cache_client(settings)
        logger.info("Cache client initialized")
    except Exception as e:
        logger.error(f"Cache client initialization failed: {e}")
        app.state.cache_client = None

    logger.info("API ready")
    yield

    logger.info("API shutdown complete")


app = FastAPI(
    title="AI Docs RAG API",
    description="AI technical documents RAG system with agentic retrieval capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
    docs_url=None,
    redoc_url="/redoc",
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with Chinese CDN."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Swagger UI",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/ask.html", include_in_schema=False)
async def ask_page():
    return FileResponse(STATIC_DIR / "ask.html")


app.include_router(ping.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(agentic_ask.router)
app.include_router(documents.router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
