# system/utils/rich_progress.py
# -*- coding: utf-8 -*-
"""
Colorful progress + tables for FCL training loops.
Hook into server.train() to get nice round-by-round visualization.

Usage (in serverbase.Server.__init__):
    from system.utils.rich_progress import RichRoundLogger
    self._roundlog = RichRoundLogger(args, fig_dir=getattr(args, "fig_dir", "figures"))

Usage (in serverbase.Server.train()):
    self._roundlog.start(total_rounds=self.global_rounds)
    ...
    self._roundlog.round_start(round_idx=i, task_id=task_id, selected_clients=selected_ids)
    ...
    self._roundlog.clients_end(round_idx=i, client_summaries=client_summaries)
    ...
    self._roundlog.round_end(round_idx=i, global_metrics={"test_acc": test_acc, "test_loss": test_loss},
                             time_cost=round_time_s)
    ...
    self._roundlog.finish()
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Optional rich UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich.rule import Rule
    from rich import box
    _RICH = True
except Exception:
    _RICH = False

def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _fmt_num(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _ascii_bar(cur: int, total: int, width: int = 30) -> str:
    cur = max(0, min(cur, total))
    filled = int(width * (cur / max(1, total)))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {cur}/{total} ({100.0*cur/max(1,total):.1f}%)"

class RichRoundLogger:
    def __init__(self, args: Any, fig_dir: str = "figures", enable: bool = True) -> None:
        self.args = args
        self.enable = enable
        self.fig_dir = _ensure_dir(fig_dir)
        self.console = Console() if _RICH else None
        self.progress: Optional[Progress] = None
        self._progress_task_id: Optional[int] = None
        self._csv_path = self.fig_dir / "round_metrics.csv"
        self._csv_initialized = False
        self._t0 = None

    # ---------- lifecycle ----------
    def start(self, total_rounds: int) -> None:
        self._t0 = time.time()
        if _RICH and self.enable:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]Global[/bold]"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[cyan]{task.percentage:>5.1f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                expand=True,
                console=self.console,
            )
            self.progress.start()
            self._progress_task_id = self.progress.add_task("rounds", total=total_rounds)
            self.console.print(Rule(style="dim"))
        else:
            print(f"[Progress] {total_rounds} global rounds.")
            print(_ascii_bar(0, total_rounds))

    def finish(self) -> None:
        if self.progress:
            try:
                self.progress.stop()
            except Exception:
                pass

    # ---------- per-round ----------
    def round_start(self, round_idx: int, task_id: Optional[int], selected_clients: Iterable[Any]) -> None:
        sc = list(selected_clients) if selected_clients is not None else []
        if _RICH and self.enable:
            header = f"[bold yellow]Round {round_idx}[/bold yellow]"
            sub = f"Task: [bold cyan]{task_id if task_id is not None else '?'}[/bold cyan] • Selected clients: [bold]{', '.join(map(str, sc)) if sc else '—'}[/bold]"
            panel = Panel(sub, title=header, border_style="yellow")
            self.console.print(panel)
        else:
            print(f"\n=== Round {round_idx} | Task {task_id if task_id is not None else '?'} ===")
            print(f"Selected clients: {', '.join(map(str, sc)) if sc else '(none)'}")

    def clients_end(self, round_idx: int, client_summaries: List[Dict[str, Any]]) -> None:
        if not client_summaries:
            return
        # Normalize accuracy and forgetting to %
        for d in client_summaries:
            acc = d.get("acc", None)
            if acc is not None and acc <= 1.0:
                d["acc"] = acc * 100.0
            forg = d.get("forg", None)                              # <<< NEW
            if forg is not None and forg <= 1.0:                    # <<< NEW
                d["forg"] = forg * 100.0                            # <<< NEW

        if _RICH and self.enable:
            table = Table(
                title=f"Local training (Round {round_idx})",
                box=box.MINIMAL_DOUBLE_HEAD,
                header_style="bold magenta",
                show_lines=False,
                expand=True,
            )
            table.add_column("Client", justify="right", style="bold")
            table.add_column("Loss", justify="right")
            table.add_column("Acc (%)", justify="right")
            table.add_column("Forgetting (%)", justify="right")      # <<< NEW
            table.add_column("Samples", justify="right")
            table.add_column("Time (s)", justify="right")

            for d in client_summaries:
                loss = d.get("loss")
                acc  = d.get("acc")
                forg = d.get("forg")                                 # <<< NEW
                samples = d.get("samples")
                tt = d.get("time")

                loss_txt = f"[bold red]{_fmt_num(loss)}[/bold red]" if loss is not None else "—"

                if acc is None:
                    acc_txt = "—"
                elif acc >= 90:
                    acc_txt = f"[bold green]{acc:.2f}[/bold green]"
                elif acc >= 70:
                    acc_txt = f"[yellow]{acc:.2f}[/yellow]"
                else:
                    acc_txt = f"[dim]{acc:.2f}[/dim]"

                # color scale for forgetting: low (good)=green, mid=yellow, high (bad)=red   # <<< NEW
                if forg is None:                                                            # <<< NEW
                    forg_txt = "—"                                                          # <<< NEW
                elif forg <= 5:                                                             # <<< NEW
                    forg_txt = f"[bold green]{forg:.2f}[/bold green]"                       # <<< NEW
                elif forg <= 15:                                                            # <<< NEW
                    forg_txt = f"[yellow]{forg:.2f}[/yellow]"                               # <<< NEW
                else:                                                                       # <<< NEW
                    forg_txt = f"[bold red]{forg:.2f}[/bold red]"                           # <<< NEW

                table.add_row(
                    str(d.get("client")),
                    loss_txt,
                    acc_txt,
                    forg_txt,                                                               # <<< NEW
                    str(samples) if samples is not None else "—",
                    _fmt_num(tt, 2) if tt is not None else "—"
                )
            self.console.print(table)
        else:
            print("Client  Loss    Acc(%)  Forget(%)  Samples  Time(s)")                     # <<< NEW
            for d in client_summaries:
                loss = _fmt_num(d.get("loss"))
                acc  = d.get("acc")
                acc_txt = "—" if acc is None else f"{(acc*100.0 if acc <= 1.0 else acc):.2f}"
                forg = d.get("forg")                                                        # <<< NEW
                forg_txt = "—" if forg is None else f"{(forg*100.0 if forg <= 1.0 else forg):.2f}"  # <<< NEW
                smp = d.get("samples")
                tt  = d.get("time")
                print(f"{d.get('client'):>6}  {loss:>6}  {acc_txt:>7}  {forg_txt:>9}  {str(smp or '—'):>7}  {_fmt_num(tt,2):>7}")  # <<< NEW


    def round_end(self, round_idx: int, global_metrics: Dict[str, Any], time_cost: Optional[float] = None) -> None:
        ta = global_metrics.get("test_acc")
        tl = global_metrics.get("test_loss")
        af = global_metrics.get("avg_forgetting_pct")              # <<< NEW: already in %
        if ta is not None and ta <= 1.0:
            ta = ta * 100.0

        # save CSV row
        self._write_csv(round_idx, ta, tl, af, time_cost)          # <<< CHANGED: pass af

        if _RICH and self.enable:
            parts = []
            if tl is not None:
                parts.append(f"Test Loss: [bold red]{_fmt_num(tl)}[/bold red]")
            if ta is not None:
                color = "bold green" if ta >= 80 else "yellow" if ta >= 60 else "dim"
                parts.append(f"Test Acc: [{color}]{ta:.2f}%[/{color}]")
            if af is not None:                                     # <<< NEW
                af_color = "bold green" if af <= 5 else "yellow" if af <= 15 else "bold red"
                parts.append(f"Avg Forget: [{af_color}]{_fmt_num(af,2)}%[/{af_color}]")
            if time_cost is not None:
                parts.append(f"Round Time: [bold cyan]{_fmt_num(time_cost,2)}s[/bold cyan]")
            msg = " • ".join(parts) if parts else "—"
            self.console.print(Panel(msg, title=f"Round {round_idx} summary", border_style="magenta"))

            if self.progress and self._progress_task_id is not None:
                self.progress.update(self._progress_task_id, advance=1)
        else:
            af_txt = f" | AvgForget={_fmt_num(af,2)}%" if af is not None else ""            # <<< NEW
            print(f"[Summary] Round {round_idx} | TestLoss={_fmt_num(tl)} | TestAcc={(f'{ta:.2f}%' if ta is not None else '—')}{af_txt} | Time={_fmt_num(time_cost,2)}s")


    # ---------- helpers ----------
    def _write_csv(self, round_idx: int, test_acc_pct: Optional[float], test_loss: Optional[float],
               avg_forgetting_pct: Optional[float], time_cost: Optional[float]) -> None:   # <<< CHANGED
        try:
            init = not self._csv_initialized or (not self._csv_path.exists())
            with open(self._csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if init:
                    w.writerow(["round", "test_acc_pct", "test_loss", "avg_forgetting_pct", "round_time_s"])  # <<< CHANGED
                    self._csv_initialized = True
                w.writerow([
                    round_idx,
                    f"{test_acc_pct:.4f}" if test_acc_pct is not None else "",
                    f"{float(test_loss):.6f}" if test_loss is not None else "",
                    f"{avg_forgetting_pct:.4f}" if avg_forgetting_pct is not None else "",                     # <<< NEW
                    f"{float(time_cost):.4f}" if time_cost is not None else ""
                ])
        except Exception:
            pass
