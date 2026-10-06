"""Streamlit page for the local-first Knowledge Agent."""

import os
import re
from typing import Any, Dict, Iterable, List

import streamlit as st


SAMPLE_QUESTIONS = [
    "3C制造企业采用了哪些管理系统？",
    "MES、QMS、ERP、PLM在制造管理中分别解决什么问题？",
    "现有知识库中有哪些与质量管理有关的方法和系统？",
]


def _has_real_tavily_key() -> bool:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    lowered = key.lower()
    return bool(key and "replace" not in lowered and "your" not in lowered)


@st.cache_resource(show_spinner=False)
def _get_agent():
    # Import lazily so the extraction page remains usable even when optional
    # Knowledge Agent dependencies are not installed.
    from knowledge_agent import build_knowledge_agent
    from knowledge_agent.factory import build_llm_from_env

    return build_knowledge_agent(build_llm_from_env())


def _cited_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[([LW]\d+)\]", answer or ""))


def _render_evidence(items: Iterable[Dict[str, Any]], empty_message: str) -> None:
    rows = list(items or [])
    if not rows:
        st.caption(empty_message)
        return
    for item in rows:
        evidence_id = item.get("id", "?")
        source_type = item.get("source_type", "unknown")
        score = item.get("score", 0)
        st.markdown(f"**[{evidence_id}] {item.get('title') or source_type}** · `{source_type}` · score={score}")
        st.write(item.get("content", ""))
        if item.get("url"):
            st.markdown(f"[打开外部来源]({item['url']})")
        st.divider()


def _render_result(result: Dict[str, Any]) -> None:
    st.subheader("问答结果")
    with st.chat_message("user"):
        st.markdown(result.get("question") or "（未记录问题）")
    with st.chat_message("assistant"):
        st.markdown(result.get("answer") or "未生成回答。")

    traces: List[Dict[str, Any]] = list(result.get("tool_trace", []))
    used_mcp = any(trace.get("protocol") == "mcp" for trace in traces)
    route = "Neo4j → Tavily MCP" if used_mcp else "Neo4j 本地检索"
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("本地证据评分", f"{float(result.get('local_score', 0)):.3f}")
    metric2.metric("实际路由", route)
    metric3.metric("Reflection 修订", int(result.get("revision_count", 0)))
    metric4.metric("降级状态", "是" if result.get("degraded") else "否")

    errors = result.get("errors", [])
    if errors:
        st.warning("部分工具调用失败，系统已使用可用证据降级回答。")
        with st.expander("查看降级原因"):
            for error in errors:
                st.code(str(error))

    answer_ids = _cited_ids(str(result.get("answer", "")))
    citations = [
        citation
        for citation in result.get("citations", [])
        if not answer_ids or citation.get("id") in answer_ids
    ]
    with st.expander("回答引用", expanded=True):
        if not citations:
            st.caption("回答没有引用可用证据。")
        for citation in citations:
            label = f"[{citation.get('id', '?')}] {citation.get('title') or '未命名来源'}"
            if citation.get("url"):
                st.markdown(f"- [{label}]({citation['url']})")
            else:
                st.markdown(f"- {label}")

    local_items = result.get("local_evidence", [])
    web_items = result.get("web_evidence", [])
    local_tab, web_tab = st.tabs([
        f"本地证据 ({len(local_items)})",
        f"外部证据 ({len(web_items)})",
    ])
    with local_tab:
        _render_evidence(local_items, "Neo4j 中没有检索到相关证据。")
    with web_tab:
        _render_evidence(web_items, "本次没有调用 Tavily MCP，或外部检索没有返回结果。")

    with st.expander("Agent 执行轨迹（面试演示）"):
        st.json(traces)


def render_knowledge_agent_page() -> None:
    st.header("Knowledge Agent 智能问答")
    st.caption("本地 Neo4j 结构化检索优先；证据不足时可通过 Tavily MCP 补充外部信息。")
    st.info("生成回答时，检索到的证据会发送给你在 .env 中配置的智谱模型。请只查询允许外发的数据。")

    if "knowledge_question" not in st.session_state:
        st.session_state.knowledge_question = SAMPLE_QUESTIONS[0]
    if "knowledge_agent_result" not in st.session_state:
        st.session_state.knowledge_agent_result = None

    st.write("示例问题")
    sample_cols = st.columns(len(SAMPLE_QUESTIONS))
    for index, question in enumerate(SAMPLE_QUESTIONS):
        if sample_cols[index].button(question, key=f"knowledge_sample_{index}", use_container_width=True):
            st.session_state.knowledge_question = question
            st.rerun()

    question = st.text_area(
        "请输入问题",
        key="knowledge_question",
        height=110,
        placeholder="例如：3C制造企业采用了哪些管理系统？",
    )

    tavily_ready = _has_real_tavily_key()
    allow_web = st.toggle(
        "本地证据不足时允许 Tavily MCP 外部检索",
        value=False,
        disabled=not tavily_ready,
        help="只有本地证据评分低于阈值时才会调用，不会每次都访问外网。",
    )
    if not tavily_ready:
        st.caption("当前未检测到可用的 TAVILY_API_KEY，因此以纯本地问答运行。配置 Key 并重启前端后即可启用。")

    run_col, clear_col = st.columns([3, 1])
    if run_col.button("向 AI 提问", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("请先输入问题。")
        else:
            try:
                with st.spinner("正在检索知识库并生成有引用的回答……"):
                    st.session_state.knowledge_agent_result = dict(
                        _get_agent().invoke(question.strip(), allow_web=allow_web)
                    )
            except Exception as exc:
                st.error(f"问答失败：{exc}")
                st.caption("请检查 .env 中的智谱、Neo4j 配置，以及相关服务是否已启动。")

    if clear_col.button("清空结果", use_container_width=True):
        st.session_state.knowledge_agent_result = None
        st.rerun()

    if st.session_state.knowledge_agent_result:
        st.divider()
        _render_result(st.session_state.knowledge_agent_result)
    else:
        st.caption("点击“向 AI 提问”后，这里会显示回答、引用证据、路由评分和工具执行轨迹。")
