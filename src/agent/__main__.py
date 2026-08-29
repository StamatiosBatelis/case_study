"""
Interactive analyst REPL.

Usage:
    python -m src.agent
    python -m src.agent --model qwen2.5:7b --base-url http://localhost:11434/v1

Type a question and press Enter. Type 'quit' or Ctrl-D to exit.
Type '/debug' before a question to print tool call trace.
"""

import argparse
import json
import sys

import kuzu
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from src.storage import kuzu_store, vector_store
from src.agent.analyst_agent import AnalystAgent

console = Console()

BANNER = """[bold cyan]Intelligence Analysis Assistant[/bold cyan]
Powered by [bold]Qwen2.5-7b[/bold] + [bold]Graph-RAG[/bold] (KuzuDB + ChromaDB)

Three systems available:
  [green]●[/green] Communications log (15 events)
  [green]●[/green] Financial transactions (12 records)
  [green]●[/green] Intelligence documents (10 reports)

Type [bold]quit[/bold] to exit. Prefix query with [bold]/debug[/bold] to see tool trace.
"""

EXAMPLE_QUERIES = [
    "Who controls Northstar Trading and what transactions did they make?",
    "Trace the money from ACC-4471 to Bluewater Ventures.",
    "What communications suggest coordination around the Rotterdam shipment?",
    "Summarise all suspicious activity linked to Marcus Vane.",
    "What is the connection between Elena Ross and Shell Corp IO?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligence analyst REPL")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model",    default="qwen2.5:7b")
    args = parser.parse_args()

    console.print(Panel(BANNER, border_style="cyan"))
    console.print("[dim]Example queries:[/dim]")
    for q in EXAMPLE_QUERIES:
        console.print(f"  [dim]• {q}[/dim]")
    console.print()

    # Open stores
    try:
        kuzu_db   = kuzu_store.open_db()
        kuzu_conn = kuzu.Connection(kuzu_db)
        chroma    = vector_store.open_client()
    except Exception as exc:
        console.print(f"[bold red]Failed to open stores: {exc}[/bold red]")
        console.print("[yellow]Run 'python -m src.pipeline' first to populate the databases.[/yellow]")
        sys.exit(1)

    agent = AnalystAgent(
        kuzu_conn=kuzu_conn,
        chroma_client=chroma,
        base_url=args.base_url,
        model=args.model,
    )

    console.print(f"[dim]Connected to {args.model} @ {args.base_url}[/dim]\n")

    while True:
        try:
            raw = console.input("[bold cyan]analyst>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        debug = raw.startswith("/debug")
        question = raw[len("/debug"):].strip() if debug else raw

        console.print()
        console.print(Rule("[dim]Investigating…[/dim]", style="dim"))

        try:
            response = agent.run(question)
        except Exception as exc:
            console.print(f"[bold red]Error: {exc}[/bold red]")
            continue

        console.print()
        console.print(Markdown(response.answer))

        if response.sources:
            console.print()
            console.print(
                "[dim]Sources:[/dim] " +
                ", ".join(sorted(set(response.sources)))
            )

        if debug and response.tool_calls_made:
            console.print()
            console.print(Rule("[dim]Tool trace[/dim]", style="dim"))
            for i, tc in enumerate(response.tool_calls_made, 1):
                console.print(
                    f"  [dim]{i}. {tc['tool']}({json_compact(tc['args'])})[/dim]"
                )

        console.print(
            f"\n[dim]({response.steps} LLM call(s), "
            f"{len(response.tool_calls_made)} tool call(s))[/dim]"
        )
        console.print()


def json_compact(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"))


if __name__ == "__main__":
    main()
