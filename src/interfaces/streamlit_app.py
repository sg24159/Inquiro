import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from config.settings import get_settings
from coordinator.graph import compile_graph


def _render_status(placeholder, running_node: str, step_start: float) -> None:
    html = (
        '<style>body{font-family:"Source Sans Pro",sans-serif;margin:0}</style>'
        f'<strong>▶ Running {running_node}…</strong> '
        '<span id="t">0.0s</span>'
        '<script>'
        f'const s={step_start * 1000:.0f};'
        'setInterval(()=>{const e=document.getElementById("t");'
        'if(e)e.textContent=((Date.now()-s)/1000).toFixed(1)+"s"},100)'
        '</script>'
    )
    with placeholder:
        components.html(html, height=50)


st.set_page_config(page_title="Inquiro", page_icon=":material/search:", layout="wide")
st.title("Inquiro — Multi-Agent Research Assistant")

if "resolved_model" not in st.session_state:
    st.session_state.resolved_model = ""
if "graph" not in st.session_state:
    st.session_state.graph = compile_graph()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "report" not in st.session_state:
    st.session_state.report = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "running" not in st.session_state:
    st.session_state.running = False
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.subheader("About")
    st.markdown(
        "Inquiro orchestrates four agents:\n"
        "- **Planner** (LLM) — decomposes query into sub-tasks\n"
        "- **Retriever** (arXiv & DDG APIs) — fetches papers by keyword\n"
        "- **Processor** (LLM) — summarises and scores findings\n"
        "- **Writer** (Jinja2) — compiles markdown + JSON assets\n\n"
        "Powered by LangGraph + Ollama."
    )
    st.divider()
    settings = get_settings()
    resolved = st.session_state.resolved_model
    if resolved:
        st.caption(f"**Model:** `{resolved}`")
    else:
        st.caption(f"**Model:** `{settings.llm_model}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input(
    "Enter your research query..." if not st.session_state.running else "Processing…",
    disabled=st.session_state.running,
)

if prompt:
    st.session_state.messages.clear()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.report = None
    st.session_state.logs = []
    st.session_state.pending = prompt
    st.session_state.running = True
    st.rerun()

elif st.session_state.running and st.session_state.pending:
    query = st.session_state.pending
    st.session_state.logs = []
    with st.chat_message("assistant"):
        report_placeholder = st.empty()
        status_placeholder = st.empty()
        log_buffer: list[tuple[str, float, str]] = []
        config = {"configurable": {"thread_id": "default"}}
        node_order = ["planner", "retriever", "processor", "writer"]
        step_start = pipeline_start = time.time()
        _render_status(status_placeholder, "planner", step_start)
        for event in st.session_state.graph.stream(
            {"query": query, "messages": [], "pipeline_start_time": pipeline_start},
            config,
            stream_mode="updates",
        ):
            now = time.time()
            step_elapsed = now - step_start
            total_elapsed = now - pipeline_start
            step_start = now
            for node, update in event.items():
                idx = node_order.index(node)
                next_node = node_order[idx + 1] if idx + 1 < len(node_order) else None
                if next_node:
                    _render_status(status_placeholder, next_node, step_start)
                else:
                    status_placeholder.empty()
                if "resolved_model" in update and update["resolved_model"]:
                    st.session_state.resolved_model = update["resolved_model"]
                if "report" in update and update["report"]:
                    st.session_state.report = update["report"]
                    report = update["report"]
                    md_path = Path(report.markdown_path)
                    body = md_path.read_text() if md_path.exists() else "# Report"
                    report_placeholder.markdown(body)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": body}
                    )
                all_logs = "\n".join(update.get("logs", []))
                if all_logs:
                    log_buffer.append((node, step_elapsed, all_logs))
        st.session_state.logs = log_buffer
    st.session_state.pending = None
    st.session_state.running = False
    st.rerun()

if st.session_state.logs:
    st.markdown("## Logs")
    for node, elapsed, text in st.session_state.logs:
        with st.expander(f"✅ `{node}` — {elapsed:.1f}s", expanded=False):
            st.code(text)

with st.sidebar:
    if st.session_state.report:
        st.subheader("Outputs")
        md = st.session_state.report.markdown_path
        js = st.session_state.report.json_path
        st.download_button("📄 Report (.md)", data=Path(md).read_text(), file_name=Path(md).name)
        st.download_button("📊 Report (.json)", data=Path(js).read_text(), file_name=Path(js).name)
