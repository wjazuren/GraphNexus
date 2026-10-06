import os
import re
from typing import Any, Dict, List, Optional


def _keywords(question: str, limit: int = 12) -> List[str]:
    latin = re.findall(r"[A-Za-z0-9_.-]{2,}", question.lower())
    chinese_segments = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    chinese: List[str] = []
    for segment in chinese_segments:
        if len(segment) <= 8:
            chinese.append(segment)
        chinese.extend(segment[index:index + 2] for index in range(max(0, len(segment) - 1)))
    seen = set()
    result = []
    for item in latin + chinese:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result[:limit]


class Neo4jStructuredRetriever:
    """Retrieve source-backed facts from the new Fact model and legacy triples."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "")
        self.username = username or os.getenv("NEO4J_USERNAME", "")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

    @property
    def configured(self) -> bool:
        return bool(self.uri and self.username and self.password)

    def retrieve(self, question: str, limit: int = 8) -> List[Dict[str, Any]]:
        if not self.configured:
            return []
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("neo4j is not installed") from exc

        keywords = _keywords(question)
        if not keywords:
            return []

        fact_query = """
        MATCH (f:Fact)-[:SUBJECT]->(h:Entity), (f)-[:OBJECT]->(t:Entity)
        WHERE any(k IN $keywords WHERE
            toLower(coalesce(h.name, '')) CONTAINS k OR
            toLower(coalesce(t.name, '')) CONTAINS k OR
            toLower(coalesce(f.predicate, '')) CONTAINS k)
        OPTIONAL MATCH (d:Document)-[:CONTAINS_FACT]->(f)
        WITH f, h, t, d,
             size([k IN $keywords WHERE
                toLower(coalesce(h.name, '')) CONTAINS k OR
                toLower(coalesce(t.name, '')) CONTAINS k OR
                toLower(coalesce(f.predicate, '')) CONTAINS k]) AS matched
        RETURN h.name AS head, f.predicate AS relation, t.name AS tail,
               coalesce(d.source, f.source, '') AS source,
               coalesce(d.id, f.source_id, '') AS source_id,
               matched AS score
        ORDER BY score DESC
        LIMIT $limit
        """
        legacy_query = """
        MATCH (h)-[r]->(t)
        WHERE NOT h:Document AND NOT h:Fact AND NOT t:Fact AND
              any(k IN $keywords WHERE
                toLower(coalesce(h.name, '')) CONTAINS k OR
                toLower(coalesce(t.name, '')) CONTAINS k OR
                toLower(coalesce(r.name, type(r))) CONTAINS k)
        WITH h, r, t,
             size([k IN $keywords WHERE
                toLower(coalesce(h.name, '')) CONTAINS k OR
                toLower(coalesce(t.name, '')) CONTAINS k OR
                toLower(coalesce(r.name, type(r))) CONTAINS k]) AS matched
        RETURN h.name AS head, coalesce(r.name, type(r)) AS relation, t.name AS tail,
               coalesce(r.source, '') AS source, coalesce(r.source_id, '') AS source_id,
               matched AS score
        ORDER BY score DESC
        LIMIT $limit
        """

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        records: List[Dict[str, Any]] = []
        try:
            with driver.session(database=self.database) as session:
                for query in (fact_query, legacy_query):
                    records.extend(dict(row) for row in session.run(
                        query, keywords=[item.lower() for item in keywords], limit=limit
                    ))
        finally:
            driver.close()

        deduplicated = {}
        for row in records:
            key = (row.get("head"), row.get("relation"), row.get("tail"))
            if key not in deduplicated or row.get("score", 0) > deduplicated[key].get("score", 0):
                deduplicated[key] = row

        evidence = []
        for index, row in enumerate(
            sorted(deduplicated.values(), key=lambda item: item.get("score", 0), reverse=True)[:limit],
            start=1,
        ):
            content = f"{row.get('head')} --{row.get('relation')}--> {row.get('tail')}"
            evidence.append({
                "id": f"L{index}",
                "source_type": "neo4j",
                "title": row.get("source") or "Neo4j structured fact",
                "url": "",
                "content": content,
                "score": float(row.get("score") or 0),
                "source_id": row.get("source_id") or "",
            })
        return evidence
