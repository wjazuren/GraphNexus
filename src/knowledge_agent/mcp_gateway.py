import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .cache import SQLiteTTLCache


@dataclass
class GatewayResult:
    evidence: List[Dict[str, Any]]
    cache_hit: bool = False
    degraded: bool = False
    error: str = ""
    latency_ms: int = 0


class TavilyMCPGateway:
    """Call Tavily's official remote MCP server with timeout and degradation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[SQLiteTTLCache] = None,
        timeout_seconds: float = 15.0,
        cache_ttl_seconds: int = 21600,
        max_attempts: int = 2,
        circuit_failure_threshold: int = 3,
        circuit_reset_seconds: int = 60,
    ):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.cache = cache or SQLiteTTLCache(os.getenv("AGENT_CACHE_PATH", ".cache/knowledge_agent.db"))
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_attempts = max_attempts
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_reset_seconds = circuit_reset_seconds
        self._failure_count = 0
        self._circuit_opened_at = 0.0

    def _circuit_is_open(self) -> bool:
        if self._failure_count < self.circuit_failure_threshold:
            return False
        if time.monotonic() - self._circuit_opened_at >= self.circuit_reset_seconds:
            self._failure_count = 0
            return False
        return True

    async def search(self, query: str, max_results: int = 5) -> GatewayResult:
        payload = {"query": query.strip(), "max_results": max_results, "search_depth": "basic"}
        cached = self.cache.get("tavily_search", payload)
        if cached is not None:
            return GatewayResult(evidence=cached, cache_hit=True)
        if not self.api_key:
            return GatewayResult([], degraded=True, error="TAVILY_API_KEY is not configured")
        if self._circuit_is_open():
            return GatewayResult([], degraded=True, error="Tavily MCP circuit breaker is open")

        started = time.perf_counter()
        last_error = ""
        for attempt in range(self.max_attempts):
            try:
                evidence = await asyncio.wait_for(
                    self._invoke_remote_mcp(payload), timeout=self.timeout_seconds
                )
                self._failure_count = 0
                self.cache.set("tavily_search", payload, evidence, self.cache_ttl_seconds)
                return GatewayResult(
                    evidence=evidence,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self.max_attempts:
                    await asyncio.sleep(0.25 * (2 ** attempt))

        self._failure_count += 1
        if self._failure_count >= self.circuit_failure_threshold:
            self._circuit_opened_at = time.monotonic()
        return GatewayResult(
            [],
            degraded=True,
            error=last_error or "Tavily MCP call failed",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _invoke_remote_mcp(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise RuntimeError("langchain-mcp-adapters is not installed") from exc

        client = MultiServerMCPClient({
            "tavily": {
                "transport": "http",
                "url": "https://mcp.tavily.com/mcp/",
                "headers": {"Authorization": f"Bearer {self.api_key}"},
            }
        })
        tools = await client.get_tools()
        search_tool = next(
            (tool for tool in tools if tool.name.replace("-", "_") == "tavily_search"), None
        )
        if search_tool is None:
            names = ", ".join(tool.name for tool in tools)
            raise RuntimeError(f"tavily_search was not exposed by the MCP server; tools={names}")
        result = await search_tool.ainvoke(payload)
        status = getattr(result, "status", "success")
        if status == "error":
            raise RuntimeError(self._content_to_text(getattr(result, "content", result)))
        return self._parse_evidence(getattr(result, "content", result))

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item)))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def _parse_evidence(self, content: Any) -> List[Dict[str, Any]]:
        text = self._content_to_text(content)
        parsed: Any = None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = content if isinstance(content, dict) else None

        rows = parsed.get("results", []) if isinstance(parsed, dict) else []
        evidence = []
        for index, row in enumerate(rows[:10], start=1):
            if not isinstance(row, dict):
                continue
            evidence.append({
                "id": f"W{index}",
                "source_type": "tavily_mcp",
                "title": row.get("title") or "Web result",
                "url": row.get("url") or "",
                "content": row.get("content") or row.get("raw_content") or "",
                "score": float(row.get("score") or 0),
            })
        if evidence:
            return evidence

        # Tavily's official MCP server formats search output as repeated
        # Title/URL/Content blocks rather than returning raw API JSON.
        block_pattern = re.compile(
            r"Title:\s*(?P<title>.*?)\n(?:ID:.*?\n)?URL:\s*(?P<url>.*?)\n"
            r"Content:\s*(?P<content>.*?)(?=\nTitle:|\nImages:|\Z)",
            re.DOTALL,
        )
        for index, match in enumerate(block_pattern.finditer(text), start=1):
            evidence.append({
                "id": f"W{index}",
                "source_type": "tavily_mcp",
                "title": match.group("title").strip(),
                "url": match.group("url").strip(),
                "content": match.group("content").strip()[:12000],
                "score": 0.5,
            })
        if evidence:
            return evidence

        urls = list(dict.fromkeys(re.findall(r"https?://[^\s\]\[<>()\"']+", text)))
        if not text.strip():
            return []
        if not urls:
            return [{
                "id": "W1",
                "source_type": "tavily_mcp",
                "title": "Tavily MCP search result",
                "url": "",
                "content": text[:12000],
                "score": 0.5,
            }]
        return [
            {
                "id": f"W{index}",
                "source_type": "tavily_mcp",
                "title": "Tavily MCP search result",
                "url": url.rstrip(".,;"),
                "content": text[:12000] if index == 1 else "",
                "score": 0.5,
            }
            for index, url in enumerate(urls[:10], start=1)
        ]
