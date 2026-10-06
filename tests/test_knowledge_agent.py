import asyncio

from src.knowledge_agent.mcp_gateway import GatewayResult, TavilyMCPGateway
from src.knowledge_agent.cache import SQLiteTTLCache
from src.knowledge_agent.workflow import KnowledgeAgent


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def get_chat_response(self, prompt):
        self.calls += 1
        if "[W1]" in prompt:
            return "外部证据补全了本地知识缺口。[W1]"
        return "该事实来自本地结构化知识。[L1]"


class FakeRetriever:
    def __init__(self, evidence):
        self.evidence = evidence

    def retrieve(self, question, limit=8):
        return self.evidence[:limit]


class FakeGateway:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def search(self, query, max_results=5):
        self.calls += 1
        return self.result


def test_local_first_path_does_not_call_mcp():
    local = [
        {
            "id": "L1",
            "source_type": "neo4j",
            "title": "doc",
            "url": "",
            "content": "OneKE 使用 Dynamic Schema 完成信息抽取",
            "score": 2,
        }
    ]
    gateway = FakeGateway(GatewayResult([]))
    agent = KnowledgeAgent(FakeLLM(), FakeRetriever(local), gateway, sufficiency_threshold=0.0)
    result = asyncio.run(agent.ainvoke("OneKE 如何完成信息抽取？"))
    assert gateway.calls == 0
    assert "[L1]" in result["answer"]
    assert result["revision_count"] == 0


def test_insufficient_local_evidence_calls_tavily_mcp_gateway():
    web = [{
        "id": "W1",
        "source_type": "tavily_mcp",
        "title": "official",
        "url": "https://example.com",
        "content": "LangGraph supports stateful agent workflows",
        "score": 0.9,
    }]
    gateway = FakeGateway(GatewayResult(web, cache_hit=True))
    agent = KnowledgeAgent(FakeLLM(), FakeRetriever([]), gateway)
    result = asyncio.run(agent.ainvoke("LangGraph 有什么能力？"))
    assert gateway.calls == 1
    assert "[W1]" in result["answer"]
    mcp_trace = next(item for item in result["tool_trace"] if item.get("protocol") == "mcp")
    assert mcp_trace["tool"] == "tavily_search"
    assert mcp_trace["cache_hit"] is True


def test_mcp_failure_degrades_without_crashing():
    gateway = FakeGateway(GatewayResult([], degraded=True, error="timeout"))
    agent = KnowledgeAgent(FakeLLM(), FakeRetriever([]), gateway)
    result = asyncio.run(agent.ainvoke("一个知识库之外的问题"))
    assert result["degraded"] is True
    assert "暂不生成事实性结论" in result["answer"]
    assert any("timeout" in error for error in result["errors"])


def test_tavily_parser_preserves_urls(tmp_path):
    gateway = TavilyMCPGateway(api_key="test", cache=SQLiteTTLCache(str(tmp_path / "parser.db")))
    content = '{"results":[{"title":"A","url":"https://example.com/a","content":"body","score":0.8}]}'
    evidence = gateway._parse_evidence(content)
    assert evidence[0]["id"] == "W1"
    assert evidence[0]["url"] == "https://example.com/a"


def test_tavily_parser_handles_official_mcp_text_format(tmp_path):
    gateway = TavilyMCPGateway(api_key="test", cache=SQLiteTTLCache(str(tmp_path / "parser.db")))
    content = """Detailed Results:

Title: Result A
URL: https://example.com/a
Content: Alpha content

Title: Result B
URL: https://example.com/b
Content: Beta content"""
    evidence = gateway._parse_evidence(content)
    assert [item["title"] for item in evidence] == ["Result A", "Result B"]
    assert evidence[1]["url"] == "https://example.com/b"
