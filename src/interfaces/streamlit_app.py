import time
import uuid
from pathlib import Path

import streamlit as st

from coordinator.graph import compile_graph

st.set_page_config(page_title="Inquiro", page_icon=":material/search:", layout="wide")
st.title("Inquiro — Multi-Agent Research Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "graph" not in st.session_state:
    st.session_state.graph = compile_graph()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "report" not in st.session_state:
    st.session_state.report = None

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.session_id, wrap_lines=True)
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.report = None
        st.rerun()

    if st.session_state.report:
        st.subheader("Outputs")
        st.markdown(f"📄 [{Path(st.session_state.report.markdown_path).name}]({st.session_state.report.markdown_path})")
        st.markdown(f"📊 [{Path(st.session_state.report.json_path).name}]({st.session_state.report.json_path})")

    st.subheader("About")
    st.markdown(
        "Inquiro orchestrates four agents:\n"
        "- **Planner** (Mistral) — decomposes query into sub-tasks\n"
        "- **Retriever** (arXiv API) — fetches papers by keyword\n"
        "- **Processor** (Mistral) — summarises and scores findings\n"
        "- **Writer** (Jinja2) — compiles markdown + JSON assets\n\n"
        "Powered by LangGraph + Ollama."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter your research query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        report_placeholder = st.empty()
        log_buffer: list[tuple[str, float, str]] = []
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        st.info("Running research pipeline…")
        last_time = time.time()
        for event in st.session_state.graph.stream(
            {"query": prompt, "messages": []}, config, stream_mode="updates"
        ):
            now = time.time()
            elapsed = now - last_time
            last_time = now
            for node, update in event.items():
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
                    log_buffer.append((node, elapsed, all_logs))
        if log_buffer:
            st.markdown("## Logs")
            for node, elapsed, text in log_buffer:
                with st.expander(f"✅ `{node}` — {elapsed:.1f}s", expanded=False):
                    st.code(text)
