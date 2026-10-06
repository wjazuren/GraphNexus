import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from knowledge_agent.mcp_gateway import TavilyMCPGateway


async def main():
    if not os.getenv("TAVILY_API_KEY"):
        raise SystemExit("Set TAVILY_API_KEY in .env before running the MCP smoke test")
    gateway = TavilyMCPGateway()
    result = await gateway.search("LangGraph official documentation", max_results=3)
    print(json.dumps({
        "ok": not result.degraded,
        "cache_hit": result.cache_hit,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "evidence": result.evidence,
    }, ensure_ascii=False, indent=2))
    if result.degraded:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

