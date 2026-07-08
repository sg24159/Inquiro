import argparse
import uuid

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from coordinator.graph import compile_graph

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Inquiro — Multi-agent research assistant"
    )
    parser.add_argument("query", type=str, help="Research query")
    parser.add_argument(
        "--session-id",
        type=str,
        default=str(uuid.uuid4()),
        help="Session ID (used in output filenames)",
    )
    args = parser.parse_args()

    graph = compile_graph()
    config = {"configurable": {"thread_id": args.session_id}}
    initial_state = {"query": args.query, "messages": []}

    console.print(f"[bold cyan]Inquiro[/bold cyan] — Session: [green]{args.session_id}[/green]\n")
    with console.status("[bold green]Researching...") as status:
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            for node, update in event.items():
                logs = update.get("logs", [])
                for log in logs:
                    console.print(f"  {log}")
                if "sub_tasks" in update:
                    tbl = Table("Sub-task", "Keywords")
                    for st in update["sub_tasks"]:
                        tbl.add_row(st.description[:60], ", ".join(st.keywords))
                    console.print(tbl)
                if "processed_findings" in update:
                    tbl = Table("Summary", "Score", "Source")
                    for pf in sorted(
                        update["processed_findings"], key=lambda x: x.relevance_score, reverse=True
                    ):
                        tbl.add_row(pf.summary[:50], f"{pf.relevance_score:.2f}", pf.source[:40])
                    console.print(tbl)
                if "report" in update and update["report"]:
                    r = update["report"]
                    console.print(f"\n[bold green]Report:[/bold green] {r.markdown_path}")
                    console.print(f"[bold green]Data:[/bold green]     {r.json_path}")

    final_state = graph.get_state(config)
    report = final_state.values.get("report")
    if report:
        md = Path(report.markdown_path).read_text() if Path(report.markdown_path).exists() else ""
        if md:
            console.print("\n[bold underline]Report Preview[/bold underline]\n")
            console.print(Markdown(md[:2000]))
    else:
        console.print("[yellow]No report was generated.[/yellow]")


if __name__ == "__main__":
    from pathlib import Path
    main()
