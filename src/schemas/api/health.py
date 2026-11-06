from typing import Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Status of an individual service."""

    status: str = Field(..., description="Service status: healthy, degraded, or unhealthy")
    message: Optional[str] = Field(None, description="Additional status details")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall service status: ok, degraded, or unhealthy")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Deployment environment")
    service_name: str = Field(..., description="Service name identifier")
    services: Dict[str, ServiceStatus] = Field(default_factory=dict, description="Individual service statuses")
