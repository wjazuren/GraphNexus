import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from knowledge_agent import build_knowledge_agent
from knowledge_agent.factory import build_llm_from_env


def percentile(values: List[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def mean(values: List[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def summarize(details: List[Dict[str, Any]], routing_enabled: bool) -> Dict[str, Any]:
    latencies = [int(item["latency_ms"]) for item in details]
    summary: Dict[str, Any] = {
        "samples": len(details),
        "citation_rate": mean([float(item["has_citation"]) for item in details]),
        "citation_grounding_rate": mean([float(item["citation_grounded"]) for item in details]),
        "expected_term_recall": mean([float(item["term_recall"]) for item in details]),
        "mcp_trigger_rate": mean([float(item["mcp_triggered"]) for item in details]),
        "degraded_rate": mean([float(item["degraded"]) for item in details]),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
    }
    route_values = [item["route_correct"] for item in details if item["route_correct"] is not None]
    summary["routing_accuracy"] = mean([float(value) for value in route_values]) if routing_enabled else None
    return summary


def escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def percent(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def render_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Knowledge Agent 评测报告",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 数据集：`{report['dataset']}`",
        f"- 联网：`{report['web_enabled']}`",
        f"- 类别过滤：`{report.get('category_filter') or 'all'}`",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 样本数 | {summary['samples']} |",
        f"| 引用率 | {percent(summary['citation_rate'])} |",
        f"| 有效引用率 | {percent(summary['citation_grounding_rate'])} |",
        f"| 预期关键词召回率 | {percent(summary['expected_term_recall'])} |",
        f"| MCP 触发率 | {percent(summary['mcp_trigger_rate'])} |",
        f"| MCP 路由准确率 | {percent(summary['routing_accuracy'])} |",
        f"| 降级率 | {percent(summary['degraded_rate'])} |",
        f"| P50 延迟 | {summary['latency_ms_p50']} ms |",
        f"| P95 延迟 | {summary['latency_ms_p95']} ms |",
        "",
        "## 分类别结果",
        "",
        "| 类别 | 样本 | 引用率 | 关键词召回率 | MCP触发率 | 路由准确率 | 降级率 | P50 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, metrics in report["by_category"].items():
        lines.append(
            f"| {escape_table(category)} | {metrics['samples']} | {percent(metrics['citation_rate'])} | "
            f"{percent(metrics['expected_term_recall'])} | {percent(metrics['mcp_trigger_rate'])} | "
            f"{percent(metrics['routing_accuracy'])} | {percent(metrics['degraded_rate'])} | "
            f"{metrics['latency_ms_p50']} ms | {metrics['latency_ms_p95']} ms |"
        )
    lines.extend([
        "",
        "## 逐题结果",
        "",
        "| ID | 类别 | 期望路由 | 实际路由 | 路由正确 | 关键词召回 | 引用 | 降级 | 延迟 |",
        "|---|---|---|---|---:|---:|---|---:|---:|",
    ])
    for item in report["details"]:
        expected_route = "MCP" if item["expect_mcp"] else "Local"
        actual_route = "MCP" if item["mcp_triggered"] else "Local"
        route_correct = "N/A" if item["route_correct"] is None else ("是" if item["route_correct"] else "否")
        lines.append(
            f"| {escape_table(item['id'])} | {escape_table(item['category'])} | {expected_route} | "
            f"{actual_route} | {route_correct} | {percent(item['term_recall'])} | "
            f"{escape_table(', '.join(item['citation_ids']) or '-')} | "
            f"{'是' if item['degraded'] else '否'} | {item['latency_ms']} ms |"
        )
    lines.extend(["", "## 回答与诊断", ""])
    for item in report["details"]:
        lines.extend([
            f"### {item['id']} · {item['question']}",
            "",
            item["answer"] or "（无回答）",
            "",
            f"- 本地分数：`{item['local_score']}`",
            f"- 命中关键词：`{item['matched_terms']}`",
            f"- 缺失关键词：`{item['missing_terms']}`",
            f"- 引用编号：`{item['citation_ids']}`",
            f"- 错误：`{item['errors']}`",
            "",
        ])
    lines.extend([
        "## 指标边界",
        "",
        "- 关键词召回率是字符串匹配，不等同于语义正确率。",
        "- 有效引用率只检查编号是否存在于证据集合，不判断证据是否真正支持结论。",
        "- 路由准确率来自测试集中的 `expect_mcp` 人工标签。",
        "- 正式写入简历前仍需人工抽查事实正确性和引用支撑性。",
        "",
    ])
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answers, citations, MCP routing and latency")
    parser.add_argument("dataset", help="JSONL with question, expected_terms and optional expect_mcp/category")
    parser.add_argument("--output", default="knowledge_agent_eval.json")
    parser.add_argument("--markdown-output", default="")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--category", default="", help="Only evaluate rows in this category")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.category:
        rows = [row for row in rows if row.get("category") == args.category]
    if not rows:
        raise SystemExit("No evaluation rows matched the requested dataset/category")

    agent = build_knowledge_agent(build_llm_from_env())
    details: List[Dict[str, Any]] = []
    routing_enabled = not args.no_web
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        result = await agent.ainvoke(row["question"], allow_web=routing_enabled)
        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = str(result.get("answer", ""))
        expected_terms = [str(term) for term in row.get("expected_terms", [])]
        matched_terms = [term for term in expected_terms if term.lower() in answer.lower()]
        missing_terms = [term for term in expected_terms if term.lower() not in answer.lower()]
        citation_ids = list(dict.fromkeys(re.findall(r"\b[LW]\d+\b", answer)))
        valid_ids = {
            str(item.get("id"))
            for item in result.get("evidence", [])
            if item.get("id")
        }
        mcp_triggered = any(
            item.get("protocol") == "mcp" for item in result.get("tool_trace", [])
        )
        expect_mcp = bool(row.get("expect_mcp", False))
        details.append({
            "id": str(row.get("id", f"Q{index:02d}")),
            "category": str(row.get("category", "uncategorized")),
            "question": row["question"],
            "answer": answer,
            "expected_terms": expected_terms,
            "matched_terms": matched_terms,
            "missing_terms": missing_terms,
            "term_recall": round(len(matched_terms) / max(1, len(expected_terms)), 4),
            "citation_ids": citation_ids,
            "has_citation": bool(citation_ids),
            "citation_grounded": bool(set(citation_ids) & valid_ids),
            "expect_mcp": expect_mcp,
            "mcp_triggered": mcp_triggered,
            "route_correct": (mcp_triggered == expect_mcp) if routing_enabled else None,
            "local_score": float(result.get("local_score", 0)),
            "needs_external": bool(result.get("needs_external", False)),
            "degraded": bool(result.get("degraded", False)),
            "revision_count": int(result.get("revision_count", 0)),
            "latency_ms": latency_ms,
            "errors": result.get("errors", []),
            "tool_trace": result.get("tool_trace", []),
        })
        print(f"[{index}/{len(rows)}] {details[-1]['id']} route={'mcp' if mcp_triggered else 'local'} latency={latency_ms}ms")

    categories = sorted({item["category"] for item in details})
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "web_enabled": routing_enabled,
        "category_filter": args.category or None,
        "summary": summarize(details, routing_enabled),
        "by_category": {
            category: summarize(
                [item for item in details if item["category"] == category],
                routing_enabled,
            )
            for category in categories
        },
        "details": details,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown_output) if args.markdown_output else output_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"JSON report: {output_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    asyncio.run(main())
