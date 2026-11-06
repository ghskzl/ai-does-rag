from fastapi import APIRouter

from ..dependencies import OpenSearchDep, SettingsDep
from ..schemas.api.health import HealthResponse, ServiceStatus
from ..services.ollama.client import OllamaClient

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: SettingsDep, opensearch_client: OpenSearchDep) -> HealthResponse:
    """Comprehensive health check endpoint for monitoring and load balancer probes.

    :returns: Service health status with version and connectivity checks
    :rtype: HealthResponse
    """
    services = {}
    overall_status = "ok"

    def _check_opensearch():
        if not opensearch_client.health_check():
            return ServiceStatus(status="unhealthy", message="Not responding")
        stats = opensearch_client.get_index_stats()
        return ServiceStatus(
            status="healthy",
            message=f"Index '{stats.get('index_name', 'unknown')}' with {stats.get('document_count', 0)} documents",
        )

    try:
        services["opensearch"] = _check_opensearch()
        if services["opensearch"].status != "healthy":
            overall_status = "degraded"
    except Exception as e:
        services["opensearch"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    try:
        ollama_client = OllamaClient(settings)
        ollama_health = await ollama_client.health_check()
        message = f"Version {ollama_health.get('version', 'unknown')}" if ollama_health["status"] == "healthy" else "Not responding"
        services["ollama"] = ServiceStatus(status=ollama_health["status"], message=message)
        if ollama_health["status"] != "healthy":
            overall_status = "degraded"
    except Exception as e:
        services["ollama"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
