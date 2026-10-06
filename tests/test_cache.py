import time

from src.knowledge_agent.cache import SQLiteTTLCache


def test_cache_hit_and_expiry(tmp_path):
    cache = SQLiteTTLCache(str(tmp_path / "cache.db"))
    payload = {"query": "OneKE"}
    cache.set("search", payload, {"ok": True}, ttl_seconds=1)
    assert cache.get("search", payload) == {"ok": True}
    time.sleep(1.05)
    assert cache.get("search", payload) is None

