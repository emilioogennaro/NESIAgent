import os
# type: ignore
import sys
import json
import time
import math
import random
import inspect
import itertools
import concurrent.futures
import threading
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.progress import (
    Progress, 
    BarColumn, 
    TextColumn, 
    TimeElapsedColumn, 
    TimeRemainingColumn, 
    SpinnerColumn
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from negmas.sao import SAOMechanism
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun

from agent.Group37_Negotiator import Group37_Negotiator
from components.acceptance import AcceptanceStrategy
from components.bidding import BiddingStrategy
from components.opponent_model import OpponentModel


# -----------------------------
# Config models
# -----------------------------

@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    n_issues: int = 2
    n_values: int = 50
    n_steps: int = 100
    reserved_min: float = 0.0
    reserved_max: float = 0.2
    domain_seed: int = 0


@dataclass(frozen=True)
class TournamentConfig:
    out_dir: str = "tournament_out"
    reps_per_pair: int = 3
    swap_sides: bool = True
    include_self_play: bool = False
    max_pairs: Optional[int] = None
    base_seed: int = 2026


@dataclass(frozen=True)
class StrategySpec:
    kind: str
    cls_name: str
    module: str
    params: Dict[str, Any]

    @property
    def label(self) -> str:
        if not self.params:
            return self.cls_name
        compact = ",".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{self.cls_name}({compact})"


@dataclass(frozen=True)
class AgentConfig:
    acceptance: StrategySpec
    bidding: StrategySpec
    opponent_model: StrategySpec

    @property
    def name(self) -> str:
        return f"A:{self.acceptance.label} | B:{self.bidding.label} | O:{self.opponent_model.label}"


# -----------------------------
# Utilities
# -----------------------------

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _bar(x: float, width: int = 18) -> str:
    x = max(0.0, min(1.0, x))
    full = int(round(x * width))
    return "█" * full + "░" * (width - full)

def _filter_kwargs_for_callable(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return accepted

def _instantiate(cls: Any, params: Dict[str, Any]) -> Any:
    filtered = _filter_kwargs_for_callable(cls, params)
    try:
        return cls(**filtered)
    except TypeError:
        return cls()

def _discover_subclasses(module_obj: Any, base_cls: Any) -> List[Any]:
    out = []
    for _, obj in inspect.getmembers(module_obj, inspect.isclass):
        if obj is base_cls:
            continue
        try:
            if inspect.isabstract(obj):
                continue
        except Exception:
            pass
        try:
            if issubclass(obj, base_cls):
                out.append(obj)
        except TypeError:
            continue
    out.sort(key=lambda c: (c.__module__, c.__name__))
    return out


# -----------------------------
# Strategy discovery
# -----------------------------

def discover_strategies() -> Tuple[List[StrategySpec], List[StrategySpec], List[StrategySpec]]:
    import components.acceptance as acc_mod
    import components.bidding as bid_mod
    import components.opponent_model as opp_mod

    acc_classes = _discover_subclasses(acc_mod, AcceptanceStrategy)
    bid_classes = _discover_subclasses(bid_mod, BiddingStrategy)
    opp_classes = _discover_subclasses(opp_mod, OpponentModel)

    acceptance_specs: List[StrategySpec] = []
    for c in acc_classes:
        if "threshold" in (inspect.signature(c).parameters if _has_signature(c) else {}):
            for thr in (0.6, 0.7, 0.8, 0.9):
                acceptance_specs.append(StrategySpec("acceptance", c.__name__, c.__module__, {"threshold": thr}))
        else:
            acceptance_specs.append(StrategySpec("acceptance", c.__name__, c.__module__, {}))

    bidding_specs: List[StrategySpec] = []
    for c in bid_classes:
        params = {}
        sig = _signature_params(c)
        if "steps" in sig:
            for s in (3, 5, 8):
                bidding_specs.append(StrategySpec("bidding", c.__name__, c.__module__, {"steps": s}))
            continue
        if "concession_exponent" in sig:
            for ce in (0.5, 1.0, 2.0):
                bidding_specs.append(StrategySpec("bidding", c.__name__, c.__module__, {"concession_exponent": ce}))
            continue
        if "time_threshold" in sig and "collapse_exponent" in sig:
            for tt in (0.6, 0.8):
                for ce in (3.0, 5.0):
                    bidding_specs.append(
                        StrategySpec("bidding", c.__name__, c.__module__, {"time_threshold": tt, "collapse_exponent": ce})
                    )
            continue
        if "strictness" in sig:
            for st in (0.8, 1.0, 1.2):
                bidding_specs.append(StrategySpec("bidding", c.__name__, c.__module__, {"strictness": st}))
            continue
        if "stall_tolerance" in sig and "unblock_bump" in sig:
            for tol in (3, 5, 8):
                for bump in (0.03, 0.05, 0.08):
                    bidding_specs.append(
                        StrategySpec("bidding", c.__name__, c.__module__, {"stall_tolerance": tol, "unblock_bump": bump})
                    )
            continue

        bidding_specs.append(StrategySpec("bidding", c.__name__, c.__module__, params))

    opponent_specs: List[StrategySpec] = []
    for c in opp_classes:
        if c.__name__ == "StrategyModel":
            for st in ("linear", "gaussian", "wavelet"):
                opponent_specs.append(
                    StrategySpec("opponent_model", c.__name__, c.__module__, {"strategy_type": st})
                )
        else:
            opponent_specs.append(StrategySpec("opponent_model", c.__name__, c.__module__, {}))

    return acceptance_specs, bidding_specs, opponent_specs

def _has_signature(callable_obj: Any) -> bool:
    try:
        inspect.signature(callable_obj)
        return True
    except (TypeError, ValueError):
        return False

def _signature_params(callable_obj: Any) -> Dict[str, Any]:
    if not _has_signature(callable_obj):
        return {}
    return dict(inspect.signature(callable_obj).parameters)


# -----------------------------
# Negotiation construction
# -----------------------------

def make_scenario_domain(s: ScenarioConfig, seed: int):
    rng = random.Random(seed)
    issues = [make_issue(name=f"{s.name}_Issue_{i}", values=s.n_values) for i in range(s.n_issues)]
    rmin = min(s.reserved_min, s.reserved_max)
    rmax = max(s.reserved_min, s.reserved_max)
    
    ufun_a = LUFun.random(issues=cast(List[Any], issues), reserved_value=(rmin, rmax))
    ufun_b = LUFun.random(issues=cast(List[Any], issues), reserved_value=(rmin, rmax))
    return issues, ufun_a, ufun_b

def build_agent(cfg: AgentConfig, name: str):
    acc_cls = getattr(__import__(cfg.acceptance.module, fromlist=[cfg.acceptance.cls_name]), cfg.acceptance.cls_name)
    bid_cls = getattr(__import__(cfg.bidding.module, fromlist=[cfg.bidding.cls_name]), cfg.bidding.cls_name)
    opp_cls = getattr(__import__(cfg.opponent_model.module, fromlist=[cfg.opponent_model.cls_name]), cfg.opponent_model.cls_name)

    opp = _instantiate(opp_cls, cfg.opponent_model.params)
    
    acc_params = cfg.acceptance.params.copy()
    acc_params['opponent_model'] = opp
    
    bid_params = cfg.bidding.params.copy()
    bid_params['opponent_model'] = opp
    
    acc = _instantiate(acc_cls, acc_params)
    bid = _instantiate(bid_cls, bid_params)

    return Group37_Negotiator(
        name=name,
        acceptance_strategy=acc,
        bidding_strategy=bid,
        opponent_model=opp,
    )

def run_one_session(
    cfg_a: AgentConfig,
    cfg_b: AgentConfig,
    scenario: ScenarioConfig,
    seed: int,
    swap_ufuns: bool,
) -> Dict[str, Any]:
    issues, ufun1, ufun2 = make_scenario_domain(scenario, seed)
    ufun_a, ufun_b = (ufun2, ufun1) if swap_ufuns else (ufun1, ufun2)

    mech = SAOMechanism(issues=issues, n_steps=scenario.n_steps)

    agent_a = build_agent(cfg_a, name="AgentA")
    agent_b = build_agent(cfg_b, name="AgentB")

    mech.add(agent_a, ufun=ufun_a)
    mech.add(agent_b, ufun=ufun_b)

    state = mech.run()
    agreement = state.agreement

    ua = float(cast(float, ufun_a(agreement))) if agreement is not None else float(cast(float, ufun_a.reserved_value))
    ub = float(cast(float, ufun_b(agreement))) if agreement is not None else float(cast(float, ufun_b.reserved_value))

    return {
        "scenario": scenario.name,
        "n_issues": scenario.n_issues,
        "n_values": scenario.n_values,
        "n_steps": scenario.n_steps,
        "seed": seed,
        "swap_ufuns": swap_ufuns,
        "cfg_a": cfg_a.name,
        "cfg_b": cfg_b.name,
        "agreement": agreement is not None,
        "steps_taken": int(getattr(state, "step", scenario.n_steps)),
        "utility_a": ua,
        "utility_b": ub,
        "welfare_sum": ua + ub,
        "fairness_absdiff": abs(ua - ub),
        "nash_product": max(0.0, ua) * max(0.0, ub),
    }


# -----------------------------
# Reporting
# -----------------------------

def make_tui_report(console: Console, df_raw: pd.DataFrame, out_dir: str) -> None:
    console.print(Panel.fit(f"[bold]Tournament finished[/bold]\nRows: {len(df_raw)}\nOut: {out_dir}", title="Status"))

    if df_raw.empty:
        console.print("[red]No results produced.[/red]")
        return

    pair_summary = (
        df_raw.groupby(["scenario", "cfg_a", "cfg_b"], as_index=False)
        .agg(
            agreement_rate=("agreement", "mean"),
            mean_ua=("utility_a", "mean"),
            mean_ub=("utility_b", "mean"),
            mean_welfare=("welfare_sum", "mean"),
            mean_fairness=("fairness_absdiff", "mean"),
            mean_nash=("nash_product", "mean"),
            mean_steps=("steps_taken", "mean"),
            n=("agreement", "size"),
        )
    )

    a_view = df_raw[["scenario", "cfg_a", "agreement", "utility_a", "steps_taken"]].copy()
    a_view.rename(columns={"cfg_a": "agent", "utility_a": "utility"}, inplace=True)
    b_view = df_raw[["scenario", "cfg_b", "agreement", "utility_b", "steps_taken"]].copy()
    b_view.rename(columns={"cfg_b": "agent", "utility_b": "utility"}, inplace=True)
    agent_rows = pd.concat([a_view, b_view], ignore_index=True)

    agent_summary = (
        agent_rows.groupby(["scenario", "agent"], as_index=False)
        .agg(
            agreement_rate=("agreement", "mean"),
            mean_utility=("utility", "mean"),
            mean_steps=("steps_taken", "mean"),
            n=("agreement", "size"),
        )
        .sort_values(["scenario", "mean_utility"], ascending=[True, False])
    )

    for scenario in sorted(df_raw["scenario"].unique()):
        top = agent_summary[agent_summary["scenario"] == scenario].head(10)
        t = Table(title=f"Top 10 agents — {scenario}", show_lines=False)
        t.add_column("Rank", justify="right")
        t.add_column("Agent", overflow="fold")
        t.add_column("Agree", justify="right")
        t.add_column("MeanU", justify="right")
        t.add_column("MeanSteps", justify="right")

        for i, row in enumerate(top.itertuples(index=False), start=1):
            agree = float(cast(float, row.agreement_rate))
            mu = float(cast(float, row.mean_utility))
            ms = float(cast(float, row.mean_steps))
            t.add_row(str(i), str(row.agent), f"{agree:0.2f} {_bar(agree)}", f"{mu:0.3f}", f"{ms:0.1f}")
        console.print(t)

    overall = (
        agent_rows.groupby(["agent"], as_index=False)
        .agg(
            agreement_rate=("agreement", "mean"),
            mean_utility=("utility", "mean"),
            mean_steps=("steps_taken", "mean"),
            n=("agreement", "size"),
        )
        .sort_values("mean_utility", ascending=False)
        .head(15)
    )

    t2 = Table(title="Overall top 15 agents (all scenarios)", show_lines=False)
    t2.add_column("Rank", justify="right")
    t2.add_column("Agent", overflow="fold")
    t2.add_column("Agree", justify="right")
    t2.add_column("MeanU", justify="right")
    t2.add_column("MeanSteps", justify="right")
    
    for i, row in enumerate(overall.itertuples(index=False), start=1):
        agree = float(cast(float, row.agreement_rate))
        meanu = float(cast(float, row.mean_utility))
        means = float(cast(float, row.mean_steps))
        t2.add_row(str(i), str(row.agent), f"{agree:0.2f} {_bar(agree)}", f"{meanu:0.3f}", f"{means:0.1f}")
    
    console.print(t2)

    pair_summary.to_csv(os.path.join(out_dir, "pair_summary.csv"), index=False)
    agent_summary.to_csv(os.path.join(out_dir, "agent_summary_by_scenario.csv"), index=False)
    overall.to_csv(os.path.join(out_dir, "agent_overall_top15.csv"), index=False)


# -----------------------------
# Main
# -----------------------------

def build_default_scenarios() -> List[ScenarioConfig]:
    return [
        ScenarioConfig(name="S_small", n_issues=2, n_values=25, n_steps=80, reserved_min=0.0, reserved_max=0.2, domain_seed=1),
        ScenarioConfig(name="S_medium", n_issues=3, n_values=50, n_steps=120, reserved_min=0.0, reserved_max=0.2, domain_seed=2),
        ScenarioConfig(name="S_large", n_issues=5, n_values=80, n_steps=160, reserved_min=0.0, reserved_max=0.2, domain_seed=3),
    ]

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Component round-robin tournament runner")
    parser.add_argument("--out", type=str, default="tournament_out", help="Output directory")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per pair per scenario")
    parser.add_argument("--no-swap", action="store_true", help="Disable swapping utility assignments")
    parser.add_argument("--self-play", action="store_true", help="Include cfg vs itself")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on number of pairs")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed")
    parser.add_argument("--speed", type=float, default=1.0, help="UI Simulation speed. 0.05=Slow, 1.0=Normal, >2.0=Hyperfast Multiprocessing")
    args = parser.parse_args()

    tcfg = TournamentConfig(
        out_dir=args.out,
        reps_per_pair=max(1, args.reps),
        swap_sides=not args.no_swap,
        include_self_play=args.self_play,
        max_pairs=args.max_pairs,
        base_seed=args.seed,
    )

    stamp_dir = os.path.join(tcfg.out_dir, _now_stamp())
    _safe_mkdir(stamp_dir)

    console = Console(record=True)

    scenarios = build_default_scenarios()
    acc_specs, bid_specs, opp_specs = discover_strategies()

    configs: List[AgentConfig] = [AgentConfig(a, b, o) for a in acc_specs for b in bid_specs for o in opp_specs]

    if tcfg.include_self_play:
        pairs = list(itertools.combinations_with_replacement(configs, 2))
    else:
        pairs = list(itertools.combinations(configs, 2))

    if tcfg.max_pairs is not None:
        pairs = pairs[: max(0, tcfg.max_pairs)]

    meta = {
        "tournament_config": asdict(tcfg),
        "scenarios": [asdict(s) for s in scenarios],
        "n_acceptance_specs": len(acc_specs),
        "n_bidding_specs": len(bid_specs),
        "n_opponent_specs": len(opp_specs),
        "n_agent_configs": len(configs),
        "n_pairs": len(pairs),
    }
    with open(os.path.join(stamp_dir, "tournament_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    console.print(Panel.fit(
        f"Configs: {len(configs)}\nPairs: {len(pairs)}\nScenarios: {len(scenarios)}\nReps/pair: {tcfg.reps_per_pair}\nHyperfast Mode: {'Yes' if args.speed > 2.0 else 'No'}",
        title="Tournament Plan Details", border_style="cyan"
    ))

    tasks: List[Tuple[AgentConfig, AgentConfig, ScenarioConfig, int, bool]] = []
    for scenario in scenarios:
        for i, (cfg_a, cfg_b) in enumerate(pairs):
            duel_seed = tcfg.base_seed + 1000000 * (hash(scenario.name) % 1000) + i * 37
            for r in range(tcfg.reps_per_pair):
                swaps = (False, True) if tcfg.swap_sides else (False,)
                for swap in swaps:
                    seed = duel_seed + 1000 * r + (1 if swap else 0)
                    tasks.append((cfg_a, cfg_b, scenario, seed, swap))

    # -----------------------------
    # Live TUI State & Components
    # -----------------------------
    
    agent_stats = {cfg.name: {"matches": 0, "utility_sum": 0.0, "agreements": 0} for cfg in configs}
    feed_messages = deque(maxlen=12) 
    feed_lock = threading.Lock()
    
    def short_name(name_str: str) -> str:
        parts = name_str.split("|")
        if len(parts) >= 2:
            return parts[1].replace("B:", "").replace("Bidding", "").strip()
        return name_str

    class LeaderboardView:
        def __rich__(self) -> Panel:
            table = Table(show_lines=False, expand=True, box=None)
            table.add_column("Rank", justify="right", style="cyan")
            table.add_column("Agent", overflow="fold")
            table.add_column("Agree", justify="right")
            table.add_column("MeanU", justify="right", style="bold green")

            leaderboard = []
            for agent, stats in agent_stats.items():
                if stats["matches"] > 0:
                    mean_u = stats["utility_sum"] / stats["matches"]
                    agree_rate = stats["agreements"] / stats["matches"]
                    leaderboard.append((mean_u, agree_rate, agent))

            leaderboard.sort(key=lambda x: x[0], reverse=True)

            for i, (mean_u, agree_rate, agent) in enumerate(leaderboard[:10], start=1):
                table.add_row(str(i), agent, f"{agree_rate:0.2f} {_bar(agree_rate, 8)}", f"{mean_u:0.3f}")
            return Panel(table, title="🏆 Live Overall Top 10 (Mean Utility)", border_style="gold1")

    class FeedView:
       def __rich__(self) -> Panel:
            table = Table(show_header=False, box=None, expand=True)
            table.add_column("Match Info")
            
            # Safely copy the deque inside the lock
            with feed_lock:
                msgs = list(feed_messages)
                
            # Iterate over the safe copy
            for msg in reversed(msgs):
                table.add_row(msg)
            return Panel(table, title="📡 Live Match Feed", border_style="blue")

    class HeaderView:
        def __rich__(self) -> Panel:
            return Panel(progress, title="Tournament Progress", border_style="green")

    layout = Layout()
    layout.split_column(Layout(name="header", size=5), Layout(name="main", ratio=1))
    layout["main"].split_row(Layout(name="leaderboard", ratio=6), Layout(name="feed", ratio=4))
    
    layout["header"].update(HeaderView())
    layout["leaderboard"].update(LeaderboardView())
    layout["feed"].update(FeedView())

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True
    )
    task_id = progress.add_task("[bold white]Running Tournament Pairs...", total=len(tasks))
    
    rows: List[Dict[str, Any]] = []

    def update_ui_state(row: Dict[str, Any]):
        for agent_key, util_key in [("cfg_a", "utility_a"), ("cfg_b", "utility_b")]:
            ag = row[agent_key]
            agent_stats[ag]["matches"] += 1
            agent_stats[ag]["utility_sum"] += row[util_key]
            if row["agreement"]:
                agent_stats[ag]["agreements"] += 1
        
        if row["agreement"]:
            if row["utility_a"] > 0.9 or row["utility_b"] > 0.9:
                status = f"🔥 [bold green]Hardball Deal! ({row['utility_a']:.2f}, {row['utility_b']:.2f})[/]"
            else:
                status = f"✅ [green]Deal ({row['utility_a']:.2f}, {row['utility_b']:.2f})[/]"
        else:
            status = "❌ [red]Timeout / Walkaway[/]"
        
        feed_text = f"[dim][{row['scenario']}][/]\n{status}\n{short_name(row['cfg_a'])} vs {short_name(row['cfg_b'])}\n"
        
        with feed_lock:
            feed_messages.append(feed_text)
            
        progress.advance(task_id, 1)

    # -----------------------------
    # Live Execution Loop
    # -----------------------------
    
    with Live(layout, console=console, refresh_per_second=12) as live:
        if args.speed > 2.0:
            # HYPERFAST MODE: Multiprocessing
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_one_session, *t) for t in tasks]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        row = future.result()
                        rows.append(row)
                        update_ui_state(row)
                    except Exception as e:
                        with feed_lock:
                            feed_messages.append(f"⚠️ [red]Match Error: {str(e)}[/red]")
                        progress.advance(task_id, 1)
        else:
            # SEQUENTIAL MODE: Artificial Delay
            delay = max(0.0, 0.1 / args.speed - 0.05)
            for t in tasks:
                row = run_one_session(*t)
                rows.append(row)
                update_ui_state(row)
                if delay > 0:
                    time.sleep(delay)

    # -----------------------------
    # Post-Tournament Execution
    # -----------------------------

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(os.path.join(stamp_dir, "raw_results.csv"), index=False)

    make_tui_report(console, df_raw, stamp_dir)

    console.save_text(os.path.join(stamp_dir, "tournament_report.txt"))
    console.print(Panel.fit(f"Wrote outputs to: {stamp_dir}", title="Done", border_style="bold green"))


if __name__ == "__main__":
    main()