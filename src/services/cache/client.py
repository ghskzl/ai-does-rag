import hashlib
import json
import logging
from datetime import timedelta
from typing import Optional

import redis

from src.config import RedisSettings

logger = logging.getLogger(__name__)


class CacheClient:
    def __init__(self, redis_client: redis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.ttl_hours)

    def _generate_cache_key(self, query: str, model: str, top_k: int, use_hybrid: bool) -> str:
        key_data = {"query": query, "model": model, "top_k": top_k, "use_hybrid": use_hybrid}
        key_hash = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
        return f"exact_cache:{key_hash}"

    async def find_cached_response(self, cache_key: str) -> Optional[str]:
        try:
            cached = self.redis.get(cache_key)
            if cached:
                return cached
            return None
        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
            return None

    async def store_response(self, cache_key: str, response_json: str) -> bool:
        try:
            return bool(self.redis.set(cache_key, response_json, ex=self.ttl))
        except Exception as e:
            logger.error(f"Cache store error: {e}")
            return False
