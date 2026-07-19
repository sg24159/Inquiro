import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from coordinator.graph import compile_graph

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Inquiro — Multi-agent research assistant"
    )
    parser.add_argument("query", type=str, help="Research query")
    args = parser.parse_args()

    graph = compile_graph()
    config = {"configurable": {"thread_id": "default"}}
    initial_state = {"query": args.query, "messages": [], "pipeline_start_time": time.time()}

    synthesized_answer = ""
    report_path = ""
    for event in graph.stream(initial_state, config, stream_mode="updates"):
        for node, update in event.items():
            logs = update.get("logs", [])
            if node == "planner" and "sub_tasks" in update:
                if logs:
                    console.print(logs[0])
                tbl = Table("Sub-task", "Keywords")
                for st in update["sub_tasks"]:
                    tbl.add_row(st.description, ", ".join(st.keywords))
                console.print(tbl)
            if node == "retriever" and logs:
                console.print(logs[0])
                for log in logs[1:]:
                    if log.startswith("    [WARN]"):
                        continue
                    console.print(f"  {log}")
            if node == "processor" and "processed_findings" in update:
                if logs:
                    console.print(logs[0])
                if update["processed_findings"]:
                    findings = sorted(
                        update["processed_findings"], key=lambda x: x.relevance_score, reverse=True
                    )
                    total = len(findings)
                    cap = min(total, 10)
                    for pf in findings[:cap]:
                        cite = ""
                        if pf.citation_author and pf.year:
                            cite = f" — {pf.citation_author} ({pf.year})"
                        elif pf.year:
                            cite = f" — ({pf.year})"
                        console.print(
                            f"  ● {pf.source}{cite} — score {pf.relevance_score}"
                        )
                    if cap < total:
                        console.print(
                            f"  [dim](showing first {cap} of {total})[/dim]"
                        )
                else:
                    console.print(
                        "  No sources met the set threshold for inclusion in the final answer."
                    )
            if "synthesized_answer" in update and update["synthesized_answer"]:
                synthesized_answer = update["synthesized_answer"]
            if "report" in update and update["report"]:
                report_path = update["report"].markdown_path

    if synthesized_answer:
        console.print(f"\n[bold]Final Answer[/bold]\n{synthesized_answer}")
    if report_path:
        console.print(f"[bold green]Report:[/bold green] {report_path}")


if __name__ == "__main__":
    main()
