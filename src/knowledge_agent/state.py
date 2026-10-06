from typing import Any, Dict, List, TypedDict


Evidence = Dict[str, Any]


class KnowledgeAgentState(TypedDict, total=False):
    question: str
    allow_web: bool
    local_evidence: List[Evidence]
    web_evidence: List[Evidence]
    evidence: List[Evidence]
    local_score: float
    needs_external: bool
    answer: str
    citations: List[Dict[str, str]]
    needs_revision: bool
    revision_count: int
    degraded: bool
    errors: List[str]
    tool_trace: List[Dict[str, Any]]

