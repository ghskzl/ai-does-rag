import logging

import redis

from src.config import Settings
from src.services.cache.client import CacheClient

logger = logging.getLogger(__name__)


def make_cache_client(settings: Settings) -> CacheClient:
    redis_settings = settings.redis
    try:
        redis_client = redis.Redis(
            host=redis_settings.host,
            port=redis_settings.port,
            password=redis_settings.password if redis_settings.password else None,
            db=redis_settings.db,
            decode_responses=redis_settings.decode_responses,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
        )
        logger.info(f"Redis client created: {redis_settings.host}:{redis_settings.port}")
        return CacheClient(redis_client, redis_settings)
    except Exception as e:
        logger.error(f"Failed to create cache client: {e}")
        raise
