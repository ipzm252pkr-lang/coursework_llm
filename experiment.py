from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box
from rich.rule import Rule

from memories.buffer_memory import BufferMemory
from memories.summary_memory import SummaryMemory
from memories.vector_memory import VectorMemory
from chatbot import Chatbot
from evaluation import Evaluator

logging.basicConfig(level=logging.WARNING)

console = Console()

CFG = {
    "buffer_max": 10,
    "summary_max": 10,
    "summary_threshold": 5,
    "vector_recent": 3,
    "vector_top_k": 4,
    "n_distractors": [0, 10, 20],
    "latency_turns": 5,
    "growth_turns": 10,
    "sleep": 6.0,
    "model": "gemini-3.1-flash-lite",
    "embedding_model": "all-MiniLM-L6-v2",
}

OUTPUT = Path("results.json")
PARTIAL = Path("results_partial.json")

COLORS = {
    "BufferMemory":  "red",
    "SummaryMemory": "blue",
    "VectorMemory":  "green",
}


def _init_client() -> genai.Client:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        console.print("[bold red]✗ GEMINI_API_KEY не знайдено у файлі .env[/bold red]")
        sys.exit(1)
    return genai.Client(api_key=key)


def _make_bots(client: genai.Client) -> dict[str, Chatbot]:
    return {
        "BufferMemory": Chatbot(
            memory=BufferMemory(max_messages=CFG["buffer_max"]),
            temperature=0.1,
        ),
        "SummaryMemory": Chatbot(
            memory=SummaryMemory(
                max_messages=CFG["summary_max"],
                llm=client,
                summarize_threshold=CFG["summary_threshold"],
            ),
            temperature=0.1,
        ),
        "VectorMemory": Chatbot(
            memory=VectorMemory(
                recent_window=CFG["vector_recent"],
                top_k=CFG["vector_top_k"],
            ),
            temperature=0.1,
        ),
    }


def _save(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "bold green"
    elif score >= 0.5:
        return "bold yellow"
    else:
        return "bold red"


def test_retention(bots: dict[str, Chatbot], ev: Evaluator) -> dict:
    console.print(Rule("[bold cyan]Тест 1 — Memory Retention[/bold cyan]"))
    results: dict = {}
    strategies = list(bots.keys())
    total = len(CFG["n_distractors"]) * len(strategies)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Виконання...", total=total)

        for n in CFG["n_distractors"]:
            results[str(n)] = {}
            for name, bot in bots.items():
                progress.update(task, description=f"[{COLORS[name]}]{name}[/] | дистрактори: {n}")
                try:
                    r = ev.run_retention_test(bot, n_distractors=n)
                    results[str(n)][name] = r.to_dict()
                except Exception as exc:
                    results[str(n)][name] = {"error": str(exc)}
                progress.advance(task)
                time.sleep(3.0)

        _save({"retention": results}, PARTIAL)

    table = Table(box=box.ROUNDED, title="Результати Memory Retention Score", title_style="bold cyan")
    table.add_column("N дистракторів", style="bold white", justify="center")
    for name in strategies:
        table.add_column(name, style=COLORS[name], justify="center")

    for n in CFG["n_distractors"]:
        row = [str(n)]
        for name in strategies:
            r = results.get(str(n), {}).get(name, {})
            s = r.get("retention_score", None)
            if isinstance(s, float):
                row.append(f"[{_score_color(s)}]{s:.4f}[/]")
            else:
                row.append("[red]ERR[/]")
        table.add_row(*row)

    console.print(table)
    return results


def test_latency(bots: dict[str, Chatbot], ev: Evaluator) -> dict:
    console.print(Rule("[bold cyan]Тест 2 - Латентність відповіді[/bold cyan]"))
    results: dict = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Виконання...", total=len(bots) * CFG["latency_turns"])

        for name, bot in bots.items():
            progress.update(task, description=f"[{COLORS[name]}]{name}[/]")
            try:
                r = ev.run_latency_test(bot, n_turns=CFG["latency_turns"])
                results[name] = r.to_dict()
            except Exception as exc:
                results[name] = {"error": str(exc)}
            for _ in range(CFG["latency_turns"]):
                progress.advance(task)
            time.sleep(3.0)

    table = Table(box=box.ROUNDED, title="Результати латентності (мс)", title_style="bold cyan")
    table.add_column("Стратегія", style="bold white")
    table.add_column("Середня", justify="center")
    table.add_column("Ст. відхилення", justify="center")
    table.add_column("Мінімум", justify="center")
    table.add_column("Максимум", justify="center")

    for name in bots:
        r = results.get(name, {})
        if "error" not in r:
            table.add_row(
                f"[{COLORS[name]}]{name}[/]",
                f"{r['mean_ms']:.0f}",
                f"{r['std_ms']:.0f}",
                f"{r['min_ms']:.0f}",
                f"{r['max_ms']:.0f}",
            )

    console.print(table)
    return results


def test_context_growth(bots: dict[str, Chatbot], ev: Evaluator) -> dict:
    console.print(Rule("[bold cyan]Тест 3 - Зростання контексту[/bold cyan]"))
    results: dict = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Виконання...", total=len(bots) * CFG["growth_turns"])

        for name, bot in bots.items():
            progress.update(task, description=f"[{COLORS[name]}]{name}[/]")
            try:
                r = ev.run_context_growth_test(bot, n_turns=CFG["growth_turns"])
                results[name] = r.to_dict()
            except Exception as exc:
                results[name] = {"error": str(exc)}
            for _ in range(CFG["growth_turns"]):
                progress.advance(task)
            time.sleep(3.0)

    table = Table(box=box.ROUNDED, title="Фінальний розмір контексту (токени)", title_style="bold cyan")
    table.add_column("Стратегія", style="bold white")
    table.add_column("Фінальний контекст", justify="center")

    for name in bots:
        r = results.get(name, {})
        if "error" not in r:
            final = r["context_sizes"][-1] if r["context_sizes"] else 0
            table.add_row(f"[{COLORS[name]}]{name}[/]", str(final))

    console.print(table)
    return results


def test_statistical(retention: dict) -> dict:
    console.print(Rule("[bold cyan]Тест 4 - Статистичний аналіз[/bold cyan]"))

    scores: dict[str, list[float]] = {}
    for n_str, strategies in retention.items():
        for name, result in strategies.items():
            if "error" not in result:
                scores.setdefault(name, []).append(result["retention_score"])

    ev = Evaluator()
    results: dict = {}

    try:
        stat_results = ev.compare_all(scores)
        for r in stat_results:
            key = f"{r.name_a}_vs_{r.name_b}"
            results[key] = r.to_dict()

        table = Table(box=box.ROUNDED, title="Результати статистичних тестів", title_style="bold cyan")
        table.add_column("Порівняння", style="bold white")
        table.add_column("Тест", justify="center")
        table.add_column("p-value", justify="center")
        table.add_column("Значуща різниця", justify="center")

        for r in stat_results:
            sig = "[bold green]Так[/]" if r.significant else "[bold red]Ні[/]"
            table.add_row(
                f"{r.name_a} vs {r.name_b}",
                r.test_name,
                f"{r.p_value:.4f}",
                sig,
            )

        console.print(table)

    except Exception as exc:
        console.print(f"[red]Статистичний тест не вдався: {exc}[/red]")
        results["error"] = str(exc)

    return results


def main() -> None:
    t0 = time.perf_counter()

    console.print(Panel(
        "[bold white]Порівняльний аналіз стратегій пам'яті для діалогових систем[/bold white]\n"
        "[dim]BufferMemory · SummaryMemory · VectorMemory[/dim]",
        title="[bold cyan]Експеримент запущено[/bold cyan]",
        border_style="cyan",
    ))

    client = _init_client()
    console.print("[bold green]✓[/bold green] API ключ знайдено")

    with console.status("[bold cyan]Ініціалізація стратегій пам'яті...[/bold cyan]"):
        bots = _make_bots(client)
    console.print("[bold green]✓[/bold green] Стратегії ініціалізовано\n")

    ev = Evaluator(llm_client=client, sleep_between_requests=CFG["sleep"])
    all_results = {"config": CFG, "retention": {}, "latency": {}, "context_growth": {}, "statistical": {}}

    all_results["retention"] = test_retention(bots, ev)
    _save(all_results, OUTPUT)

    all_results["latency"] = test_latency(bots, ev)
    _save(all_results, OUTPUT)

    all_results["context_growth"] = test_context_growth(bots, ev)
    _save(all_results, OUTPUT)

    all_results["statistical"] = test_statistical(all_results["retention"])
    _save(all_results, OUTPUT)

    elapsed = (time.perf_counter() - t0) / 60

    console.print(Panel(
        f"[bold green]Експеримент завершено за {elapsed:.1f} хвилин[/bold green]\n"
        f"Результати збережено у [bold white]{OUTPUT}[/bold white]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
