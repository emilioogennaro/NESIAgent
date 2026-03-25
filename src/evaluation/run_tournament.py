import os
# type: ignore
import sys
import json
import hashlib
import time
import math
import random
import inspect
import itertools
import statistics
import concurrent.futures
import threading
import importlib
import csv
import subprocess
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

RAM_HARD_LIMIT_GB: Optional[int] = None

import pandas as pd
from rich.console import Console, Group
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
    SpinnerColumn,
    TaskID,
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

def _stable_hash_mod(text: str, mod: int) -> int:
    if mod <= 0:
        raise ValueError("mod must be > 0")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % mod

def _get_total_ram_gb() -> Optional[int]:
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            bytes_total = int(result.stdout.strip())
        else:
            bytes_total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        return max(1, int(bytes_total // (1024 ** 3)))
    except Exception:
        return None

def _get_process_rss_gb(pid: int) -> Optional[float]:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        rss_kb = int(result.stdout.strip())
        return rss_kb / 1024 / 1024
    except Exception:
        return None

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
    cfg_a: Any,
    cfg_b: Any,
    scenario: ScenarioConfig,
    seed: int,
    swap_ufuns: bool,
) -> Dict[str, Any]:
    issues, ufun1, ufun2 = make_scenario_domain(scenario, seed)
    ufun_a, ufun_b = (ufun2, ufun1) if swap_ufuns else (ufun1, ufun2)

    mech = SAOMechanism(issues=issues, n_steps=scenario.n_steps)

    def resolve_agent(cfg, name: str):
        return build_agent(cfg, name=name)


    agent_a = resolve_agent(cfg_a, "AgentA")
    agent_b = resolve_agent(cfg_b, "AgentB")

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
# Bandit Statistics
# -----------------------------

def get_ucb_lcb(utilities: List[float], c_value: float) -> Tuple[float, float, float, int]:
    n = len(utilities)
    if n == 0:
        return 1.0, 0.0, 0.5, 0  # UCB, LCB, Mean, Count
    
    mu = sum(utilities) / n
    if n < 2:
        sigma = 0.5
    else:
        sigma = statistics.stdev(utilities)
        
    margin = c_value * (sigma / math.sqrt(n))
    return mu + margin, mu - margin, mu, n


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
    parser.add_argument("--speed", type=float, default=1.0, help="UI Simulation speed. 0.05=Slow, >2.0=Multiprocessing")
    parser.add_argument("--workers", type=int, default=4, help="Max worker processes in multiprocessing mode")
    parser.add_argument("--rss-limit-gb", type=int, default=None, help="Hard RSS cap for this process (GB)")
    parser.add_argument("--s-large", action="store_true", help="Enable larger scenario (5 issues, 80 values, 160 steps)")
    
    # Dynamic Tournament Arguments
    parser.add_argument("--dynamic", action="store_true", help="Use Swiss-Bandit dynamic tournament scheduling")
    parser.add_argument("--grace-matches", type=int, default=5, help="Grace matches per agent before pruning (Dynamic only)")
    parser.add_argument("--swiss-rounds", type=int, default=5, help="Number of adaptive Swiss rounds per scenario (Dynamic only)")
    parser.add_argument("--dynamic-reps", type=int, default=1, help="Repetitions per dynamic pairing (Dynamic only)")
    parser.add_argument("--c-value", type=float, default=1.96, help="Confidence interval multiplier for pruning (Dynamic only)")
    parser.add_argument("--scene-pause", type=float, default=0.8, help="Pause in seconds between UI scene transitions")
    
    args = parser.parse_args()
    dynamic_reps = max(1, args.dynamic_reps)

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

    console = Console(record=False)

    scenarios = build_default_scenarios()

    if args.s_large:
        scenarios.append(ScenarioConfig(name="S_large", n_issues=5, n_values=80, n_steps=160, reserved_min=0.0, reserved_max=0.2, domain_seed=3))

    acc_specs, bid_specs, opp_specs = discover_strategies()

    opp_spec_lines = "\n".join(f"• {spec.label}" for spec in opp_specs)
    console.print(Panel.fit(
        f"Discovered Opponent Model Specs ({len(opp_specs)}):\n{opp_spec_lines}",
        title="Opponent-Model Coverage", border_style="magenta"
    ))

    opponent_aware_acceptance = {"OpponentAwareAcceptance"}
    opponent_aware_bidding = {"OpponentAwareBidding"}

    no_opponent_model = next((spec for spec in opp_specs if spec.cls_name == "NoOpponentModel"), None)
    if no_opponent_model is None and opp_specs:
        no_opponent_model = opp_specs[0]

    configs: List[Any] = []
    for a in acc_specs:
        for b in bid_specs:
            is_opponent_aware = (a.cls_name in opponent_aware_acceptance) or (b.cls_name in opponent_aware_bidding)
            if is_opponent_aware:
                for o in opp_specs:
                    configs.append(AgentConfig(a, b, o))
            elif no_opponent_model is not None:
                configs.append(AgentConfig(a, b, no_opponent_model))

    meta = {
        "tournament_config": asdict(tcfg),
        "scenarios": [asdict(s) for s in scenarios],
        "n_acceptance_specs": len(acc_specs),
        "n_bidding_specs": len(bid_specs),
        "n_opponent_specs": len(opp_specs),
        "n_agent_configs": len(configs),
        "dynamic_mode": args.dynamic,
        "dynamic_reps": dynamic_reps,
    }
    with open(os.path.join(stamp_dir, "tournament_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    max_pairs_label = str(tcfg.max_pairs) if tcfg.max_pairs is not None else "None (full)"
    swaps_per_pair = 2 if tcfg.swap_sides else 1
    dynamic_mode_label = "Yes" if args.dynamic else "No"
    plan_details = (
        f"Configs: {len(configs)}\n"
        f"  - Acceptance specs: {len(acc_specs)}\n"
        f"  - Bidding specs: {len(bid_specs)}\n"
        f"  - Opponent specs: {len(opp_specs)}\n"
        f"Scenarios: {len(scenarios)} ({', '.join(s.name for s in scenarios)})\n"
        f"Base Seed: {tcfg.base_seed}\n"
        f"Dynamic Scheduling: {dynamic_mode_label}\n"
        f"Self Play: {'Yes' if tcfg.include_self_play else 'No'}\n"
        f"Swap Sides: {'Yes' if tcfg.swap_sides else 'No'} (x{swaps_per_pair})\n"
        f"Static Reps/Pair: {tcfg.reps_per_pair}\n"
        f"Max Pairs: {max_pairs_label}\n"
        f"Dynamic Grace Matches: {args.grace_matches}\n"
        f"Dynamic Swiss Rounds: {args.swiss_rounds}\n"
        f"Dynamic Reps/Pairing: {dynamic_reps}\n"
        f"Dynamic C-Value: {args.c_value}\n"
        f"Scene Pause: {args.scene_pause}s\n"
        f"Hyperfast Mode: {'Yes' if args.speed > 2.0 else 'No'} (speed={args.speed})"
    )
    console.print(Panel.fit(plan_details, title="Tournament Plan Details", border_style="cyan"))

    def pause_scene_transition() -> None:
        if args.scene_pause > 0:
            time.sleep(args.scene_pause)

    pause_scene_transition()

    n_configs = len(configs)
    full_pairings_per_scenario = (
        (n_configs * (n_configs + 1)) // 2 if tcfg.include_self_play else (n_configs * (n_configs - 1)) // 2
    )
    full_pairings_total = full_pairings_per_scenario * len(scenarios)
    pairing_task_multiplier = swaps_per_pair * (dynamic_reps if args.dynamic else tcfg.reps_per_pair)

    # -----------------------------
    # Live TUI State & Components
    # -----------------------------
    
    agent_stats = {cfg.name: {"matches": 0, "utility_sum": 0.0, "agreements": 0} for cfg in configs}
    feed_messages = deque(maxlen=12) 
    feed_lock = threading.Lock()
    runtime_state = {
        "mode": "Dynamic" if args.dynamic else "Static",
        "scenario": "-",
        "phase": "Initializing",
        "alive_agents": len(configs),
        "total_agents": len(configs),
        "start_ts": time.time(),
        "batches_run": 0,
        "tasks_submitted": 0,
        "completed_matches": 0,
        "completed_agreements": 0,
        "completed_failures": 0,
        "hardball_deals": 0,
        "sum_welfare": 0.0,
        "sum_fairness": 0.0,
        "sum_nash": 0.0,
        "sum_steps": 0.0,
        "last_result": "-",
        "last_matchup": "-",
        "pruned_total": 0,
        "last_pruned": 0,
        "current_round": "-",
    }
    phase_index = 0
    phase_total = 0
    
    def short_name(name_str: str) -> str:
        def compact_component(component: str) -> str:
            value = component.strip()
            for suffix in ("AcceptanceStrategy", "BiddingStrategy", "OpponentModel", "Strategy", "Model"):
                if value.endswith(suffix):
                    value = value[: -len(suffix)]
                    break
            replacements = {
                "concession_exponent": "ce",
                "time_threshold": "tt",
                "collapse_exponent": "cx",
                "stall_tolerance": "st",
                "unblock_bump": "ub",
                "strategy_type": "type",
                "strictness": "str",
                "threshold": "thr",
                "steps": "s",
            }
            for old, new in replacements.items():
                value = value.replace(old, new)
            return value

        parts = name_str.split("|")
        if len(parts) >= 3:
            acceptance = compact_component(parts[0].replace("A:", "").strip())
            bidding = compact_component(parts[1].replace("B:", "").strip())
            opponent = compact_component(parts[2].replace("O:", "").strip())
            return f"A:{acceptance} | B:{bidding} | O:{opponent}"
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
                table.add_row(str(i), short_name(agent), f"{agree_rate:0.2f} {_bar(agree_rate, 8)}", f"{mean_u:0.3f}")

            return Panel(table, title="🏆 Live Overall Top 10 (Mean Utility)", border_style="gold1")

    overall_progress = Progress(
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
    phase_progress = Progress(
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

    class StatsView:
        def __rich__(self) -> Panel:
            leaderboard = []
            for agent, stats in agent_stats.items():
                if stats["matches"] > 0:
                    mean_u = stats["utility_sum"] / stats["matches"]
                    agree_rate = stats["agreements"] / stats["matches"]
                    leaderboard.append((mean_u, agree_rate, agent))

            leaderboard.sort(key=lambda x: x[0], reverse=True)

            total_agent_entries = sum(int(stats["matches"]) for stats in agent_stats.values())
            completed_matches = total_agent_entries // 2
            agreement_entries = sum(int(stats["agreements"]) for stats in agent_stats.values())
            completed_agreements = agreement_entries // 2
            global_agree_rate = (completed_agreements / completed_matches) if completed_matches > 0 else 0.0
            global_mean_u = (
                sum(float(stats["utility_sum"]) for stats in agent_stats.values()) / total_agent_entries
                if total_agent_entries > 0
                else 0.0
            )

            elapsed = max(1e-9, time.time() - float(runtime_state["start_ts"]))
            throughput = float(runtime_state["completed_matches"]) / elapsed
            mean_welfare = (
                float(runtime_state["sum_welfare"]) / float(runtime_state["completed_matches"])
                if runtime_state["completed_matches"] > 0
                else 0.0
            )
            mean_fairness = (
                float(runtime_state["sum_fairness"]) / float(runtime_state["completed_matches"])
                if runtime_state["completed_matches"] > 0
                else 0.0
            )
            mean_nash = (
                float(runtime_state["sum_nash"]) / float(runtime_state["completed_matches"])
                if runtime_state["completed_matches"] > 0
                else 0.0
            )
            mean_steps = (
                float(runtime_state["sum_steps"]) / float(runtime_state["completed_matches"])
                if runtime_state["completed_matches"] > 0
                else 0.0
            )

            active_task = phase_progress.tasks[-1] if len(phase_progress.tasks) > 0 else None
            task_done = int(active_task.completed) if active_task is not None else 0
            task_total = int(active_task.total) if (active_task is not None and active_task.total is not None) else 0
            task_pct = (task_done / task_total) if task_total > 0 else 0.0
            best_agent_label = short_name(leaderboard[0][2]) if len(leaderboard) > 0 else "-"
            best_agent_mu = leaderboard[0][0] if len(leaderboard) > 0 else 0.0
            pairings_explored = int(runtime_state["tasks_submitted"]) // max(1, pairing_task_multiplier)
            pairings_coverage = (pairings_explored / full_pairings_total) if full_pairings_total > 0 else 0.0

            stats_table = Table(show_header=False, box=None, expand=True, pad_edge=False)
            stats_table.add_column("k", style="bold cyan", no_wrap=True)
            stats_table.add_column("v", overflow="fold")
            stats_table.add_row("Mode", str(runtime_state["mode"]))
            stats_table.add_row("Scenario", str(runtime_state["scenario"]))
            stats_table.add_row("Phase", str(runtime_state["phase"]))
            stats_table.add_row("Round", str(runtime_state["current_round"]))
            stats_table.add_row(
                "Alive Agents",
                f"{runtime_state['alive_agents']}/{runtime_state['total_agents']}",
            )
            stats_table.add_row("Agents Scored", f"{len(leaderboard)}/{len(configs)}")
            stats_table.add_row("Full Pairings/S", f"{full_pairings_per_scenario}")
            stats_table.add_row("Full Pairings Tot", f"{full_pairings_total}")
            stats_table.add_row("Pairings Explored", f"{pairings_explored}")
            stats_table.add_row("Pairings Coverage", f"{pairings_coverage:0.2%}")
            stats_table.add_row("Completed Matches", f"{completed_matches}")
            stats_table.add_row("Global Agree", f"{global_agree_rate:0.2f} {_bar(global_agree_rate, 8)}")
            stats_table.add_row("Global MeanU", f"{global_mean_u:0.3f}")
            stats_table.add_row("Failures", f"{runtime_state['completed_failures']}")
            stats_table.add_row("Hardball Deals", f"{runtime_state['hardball_deals']}")
            stats_table.add_row("Mean Welfare", f"{mean_welfare:0.3f}")
            stats_table.add_row("Mean Fairness", f"{mean_fairness:0.3f}")
            stats_table.add_row("Mean Nash", f"{mean_nash:0.3f}")
            stats_table.add_row("Mean Steps", f"{mean_steps:0.1f}")
            stats_table.add_row("Batches Run", f"{runtime_state['batches_run']}")
            stats_table.add_row("Tasks Submitted", f"{runtime_state['tasks_submitted']}")
            stats_table.add_row("Task Progress", f"{task_done}/{task_total} ({task_pct:0.1%})")
            stats_table.add_row("Elapsed", f"{elapsed:0.1f}s")
            stats_table.add_row("Throughput", f"{throughput:0.2f} matches/s")
            stats_table.add_row("Pruned Total", f"{runtime_state['pruned_total']}")
            stats_table.add_row("Best Agent", f"{best_agent_label} ({best_agent_mu:0.3f})")
            stats_table.add_row("Last Result", str(runtime_state["last_result"]))

            return Panel(stats_table, title="📊 Live Stats", border_style="cyan")

    class FeedView:
       def __rich__(self) -> Panel:
            table = Table(show_header=False, box=None, expand=True)
            table.add_column("Match Info")
            
            with feed_lock:
                msgs = list(feed_messages)
                
            for msg in reversed(msgs):
                table.add_row(msg)
            return Panel(table, title="📡 Live Match Feed", border_style="blue")

    class HeaderView:
        def __rich__(self) -> Panel:
            return Panel(
                Group(overall_progress, phase_progress),
                title="Tournament Progress",
                border_style="green",
            )

    layout = Layout()
    layout.split_column(Layout(name="header", size=5), Layout(name="main", ratio=1))
    layout["main"].split_row(Layout(name="left", ratio=6), Layout(name="feed", ratio=4))
    layout["left"].split_column(Layout(name="leaderboard", ratio=2), Layout(name="stats", ratio=3))
    
    layout["header"].update(HeaderView())
    layout["leaderboard"].update(LeaderboardView())
    layout["stats"].update(StatsView())
    layout["feed"].update(FeedView())

    raw_csv_path = os.path.join(stamp_dir, "raw_results.csv")
    raw_file = open(raw_csv_path, "w", encoding="utf-8", newline="")
    raw_writer: Dict[str, Any] = {"writer": None}

    def write_row(row: Dict[str, Any]) -> None:
        writer = raw_writer["writer"]
        if writer is None:
            writer = csv.DictWriter(raw_file, fieldnames=list(row.keys()))
            writer.writeheader()
            raw_writer["writer"] = writer
        writer.writerow(row)
        if runtime_state["completed_matches"] % 100 == 0:
            raw_file.flush()

    def update_ui_state(row: Dict[str, Any], task_id: TaskID):
        runtime_state["completed_matches"] += 1
        runtime_state["sum_welfare"] += float(row["welfare_sum"])
        runtime_state["sum_fairness"] += float(row["fairness_absdiff"])
        runtime_state["sum_nash"] += float(row["nash_product"])
        runtime_state["sum_steps"] += float(row["steps_taken"])
        runtime_state["last_matchup"] = f"{short_name(row['cfg_a'])} vs {short_name(row['cfg_b'])}"

        for agent_key, util_key in [("cfg_a", "utility_a"), ("cfg_b", "utility_b")]:
            ag = row[agent_key]
            agent_stats[ag]["matches"] += 1
            agent_stats[ag]["utility_sum"] += row[util_key]
            if row["agreement"]:
                agent_stats[ag]["agreements"] += 1
        
        if row["agreement"]:
            runtime_state["completed_agreements"] += 1
            if row["utility_a"] > 0.9 or row["utility_b"] > 0.9:
                runtime_state["hardball_deals"] += 1
                status = f"🔥 [bold green]Hardball Deal! ({row['utility_a']:.2f}, {row['utility_b']:.2f})[/]"
            else:
                status = f"✅ [green]Deal ({row['utility_a']:.2f}, {row['utility_b']:.2f})[/]"
            runtime_state["last_result"] = f"Deal ({row['utility_a']:.2f}, {row['utility_b']:.2f})"
        else:
            runtime_state["completed_failures"] += 1
            status = "❌ [red]Timeout / Walkaway[/]"
            runtime_state["last_result"] = "Timeout / Walkaway"
        
        feed_text = f"[dim][{row['scenario']}][/]\n{status}\n{short_name(row['cfg_a'])} vs {short_name(row['cfg_b'])}\n"
        
        with feed_lock:
            feed_messages.append(feed_text)
            
        if phase_task_id is not None:
            phase_progress.advance(phase_task_id, 1)
            completed = float(phase_progress.tasks[phase_task_id].completed or 0)
            total = float(phase_progress.tasks[phase_task_id].total or 0)
            overall_completed = float(phase_index) + (completed / total if total > 0 else 0.0)
            overall_progress.update(task_id, completed=overall_completed)

    def run_task_batch(
        tasks_batch: List[Tuple],
        executor,
        task_id: TaskID,
        scenario_utils: Optional[Dict[str, List[float]]],
        extend_total: bool,
    ):
        if shutdown_event.is_set():
            return
        runtime_state["batches_run"] += 1
        runtime_state["tasks_submitted"] += len(tasks_batch)
        if extend_total:
            existing_total = overall_progress.tasks[task_id].total
            overall_progress.update(task_id, total=(existing_total or 0) + len(tasks_batch))
        if executor:
            futures = [executor.submit(run_one_session, *t) for t in tasks_batch]
            pending = set(futures)
            while pending and not shutdown_event.is_set():
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    try:
                        row = future.result(timeout=0)
                        write_row(row)

                        if scenario_utils is not None:
                            scenario_utils[row["cfg_a"]].append(row["utility_a"])
                            scenario_utils[row["cfg_b"]].append(row["utility_b"])

                        update_ui_state(row, task_id)

                    except concurrent.futures.TimeoutError:
                        feed_messages.append("⏰ [red]Match Timeout[/red]")
                        if phase_task_id is not None:
                            phase_progress.advance(phase_task_id, 1)
                            completed = float(phase_progress.tasks[phase_task_id].completed or 0)
                            total = float(phase_progress.tasks[phase_task_id].total or 0)
                            overall_completed = float(phase_index) + (completed / total if total > 0 else 0.0)
                            overall_progress.update(task_id, completed=overall_completed)

                    except Exception as e:
                        feed_messages.append(f"⚠️ [red]Match Error: {str(e)}[/red]")
                        if phase_task_id is not None:
                            phase_progress.advance(phase_task_id, 1)
                            completed = float(phase_progress.tasks[phase_task_id].completed or 0)
                            total = float(phase_progress.tasks[phase_task_id].total or 0)
                            overall_completed = float(phase_index) + (completed / total if total > 0 else 0.0)
                            overall_progress.update(task_id, completed=overall_completed)

            if shutdown_event.is_set() and pending:
                for future in pending:
                    future.cancel()
        else:
            delay = max(0.0, 0.1 / args.speed - 0.05)
            for t in tasks_batch:
                if shutdown_event.is_set():
                    return
                row = run_one_session(*t)
                write_row(row)
                if scenario_utils is not None:
                    scenario_utils[row["cfg_a"]].append(row["utility_a"])
                    scenario_utils[row["cfg_b"]].append(row["utility_b"])
                update_ui_state(row, task_id)
                if delay > 0:
                    time.sleep(delay)

    # -----------------------------
    # Live Execution Loop
    # -----------------------------
    
    with Live(layout, console=console, refresh_per_second=12) as live:
        shutdown_event = threading.Event()
        phase_task_id: Optional[TaskID] = None
        overall_task_id: Optional[TaskID] = None
        max_workers = max(1, args.workers)
        rss_limit_gb = args.rss_limit_gb if args.rss_limit_gb is not None else RAM_HARD_LIMIT_GB
        if rss_limit_gb is not None:
            def _monitor_rss() -> None:
                while not shutdown_event.is_set():
                    rss_gb = _get_process_rss_gb(os.getpid())
                    if rss_gb is not None and rss_gb >= rss_limit_gb:
                        shutdown_event.set()
                        console.print(Panel.fit(
                            f"RAM limit reached ({rss_gb:0.1f} GB >= {rss_limit_gb} GB). Stopping.",
                            title="Memory Guard",
                            border_style="red",
                        ))
                        return
                    time.sleep(1.0)
            threading.Thread(target=_monitor_rss, daemon=True).start()
        executor = (
            concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            if args.speed > 2.0
            else None
        )
        
        if not args.dynamic:
            runtime_state["phase"] = "Static Round Robin"
            runtime_state["alive_agents"] = len(configs)
            runtime_state["current_round"] = "-"
            if tcfg.include_self_play:
                pairs = list(itertools.combinations_with_replacement(configs, 2))
            else:
                pairs = list(itertools.combinations(configs, 2))

            if tcfg.max_pairs is not None:
                pairs = pairs[: max(0, tcfg.max_pairs)]

            total_tasks = 0
            for scenario in scenarios:
                n_pairs = len(pairs)
                total_tasks += n_pairs * tcfg.reps_per_pair * swaps_per_pair

            overall_task_id = overall_progress.add_task("[bold white]Overall Progress", total=len(scenarios))
            phase_index = 0
            for scenario in scenarios:
                if shutdown_event.is_set():
                    break
                runtime_state["scenario"] = scenario.name
                runtime_state["phase"] = "Static Scenario"
                feed_messages.append(f"📍 [bold cyan]Static Scenario: {scenario.name}[/]")
                pause_scene_transition()
                tasks = []
                for i, (cfg_a, cfg_b) in enumerate(pairs):
                    duel_seed = tcfg.base_seed + 1000000 * _stable_hash_mod(scenario.name, 1000) + i * 37
                    for r in range(tcfg.reps_per_pair):
                        swaps = (False, True) if tcfg.swap_sides else (False,)
                        for swap in swaps:
                            seed = duel_seed + 1000 * r + (1 if swap else 0)
                            tasks.append((cfg_a, cfg_b, scenario, seed, swap))
                phase_total = len(tasks)
                if phase_task_id is None:
                    phase_task_id = phase_progress.add_task(
                        f"[bold white]Scenario {scenario.name}",
                        total=phase_total,
                    )
                else:
                    phase_progress.reset(
                        phase_task_id,
                        description=f"[bold white]Scenario {scenario.name}",
                        total=phase_total,
                    )
                run_task_batch(tasks, executor, overall_task_id, None, False)
                phase_index += 1
                overall_progress.update(overall_task_id, completed=float(phase_index))
                pause_scene_transition()
            
        else:
            # DYNAMIC SWISS-BANDIT SYSTEM
            rng = random.Random(tcfg.base_seed)
            active_configs = configs[:]
            runtime_state["phase"] = "Dynamic Init"
            runtime_state["alive_agents"] = len(active_configs)
            runtime_state["current_round"] = "Init"
            total_phases = len(scenarios) * (1 + args.swiss_rounds)
            overall_task_id = overall_progress.add_task("[bold white]Overall Progress", total=total_phases)
            phase_index = 0
            
            for scenario in scenarios:
                if shutdown_event.is_set():
                    break
                if len(active_configs) < 2:
                    break
                
                runtime_state["scenario"] = scenario.name
                runtime_state["phase"] = "Scenario Escalation"
                runtime_state["alive_agents"] = len(active_configs)
                runtime_state["current_round"] = "Escalation"
                feed_messages.append(f"🏁 [bold magenta]Escalating to Scenario: {scenario.name}[/]")
                pause_scene_transition()
                scenario_utils = {c.name: [] for c in active_configs}
                batch_size = 200
                
                runtime_state["phase"] = "Grace Period"
                runtime_state["current_round"] = "Grace"
                feed_messages.append(f"⚖️ [blue]Grace Period ({args.grace_matches} matches each)[/]")
                pause_scene_transition()
                pairings = (len(active_configs) // 2) + (1 if len(active_configs) % 2 != 0 else 0)
                phase_total = args.grace_matches * pairings * swaps_per_pair * dynamic_reps
                if phase_task_id is None:
                    phase_task_id = phase_progress.add_task(
                        f"[bold white]Grace {scenario.name}",
                        total=phase_total,
                    )
                else:
                    phase_progress.reset(
                        phase_task_id,
                        description=f"[bold white]Grace {scenario.name}",
                        total=phase_total,
                    )
                grace_tasks: List[Tuple] = []
                for _ in range(args.grace_matches):
                    if shutdown_event.is_set():
                        break
                    shuffled = active_configs[:]
                    rng.shuffle(shuffled)
                    
                    for i in range(0, len(shuffled)-1, 2):
                        c1, c2 = shuffled[i], shuffled[i+1]
                        swaps = (False, True) if tcfg.swap_sides else (False,)
                        for _rep in range(dynamic_reps):
                            for swap in swaps:
                                grace_tasks.append((c1, c2, scenario, rng.randint(0, 999999), swap))
                                if len(grace_tasks) >= batch_size:
                                    run_task_batch(grace_tasks, executor, overall_task_id, scenario_utils, False)
                                    grace_tasks.clear()
                            
                    if len(shuffled) % 2 != 0:
                        c1, c2 = shuffled[-1], rng.choice(shuffled[:-1])
                        swaps = (False, True) if tcfg.swap_sides else (False,)
                        for _rep in range(dynamic_reps):
                            for swap in swaps:
                                grace_tasks.append((c1, c2, scenario, rng.randint(0, 999999), swap))
                                if len(grace_tasks) >= batch_size:
                                    run_task_batch(grace_tasks, executor, overall_task_id, scenario_utils, False)
                                    grace_tasks.clear()

                if grace_tasks:
                    run_task_batch(grace_tasks, executor, overall_task_id, scenario_utils, False)
                phase_index += 1
                overall_progress.update(overall_task_id, completed=float(phase_index))
                pause_scene_transition()

                for round_idx in range(args.swiss_rounds):
                    if shutdown_event.is_set():
                        break
                    if len(active_configs) < 2:
                        break
                        
                    stats = {c.name: get_ucb_lcb(scenario_utils[c.name], args.c_value) for c in active_configs}
                    max_lcb = max(s[1] for s in stats.values())
                    
                    survivors = []
                    pruned_this_round = 0
                    for c in active_configs:
                        ucb, lcb, mu, n = stats[c.name]
                        if ucb >= max_lcb or (len(survivors) < 2 and len(active_configs) <= 2):
                            survivors.append(c)
                        else:
                            pruned_this_round += 1
                            feed_messages.append(f"✂️ [red]Pruned[/] {short_name(c.name)} (UCB:{ucb:.2f} < {max_lcb:.2f})")
                            
                    active_configs = survivors
                    runtime_state["alive_agents"] = len(active_configs)
                    runtime_state["last_pruned"] = pruned_this_round
                    runtime_state["pruned_total"] += pruned_this_round
                    if len(active_configs) < 2:
                        break
                        
                    runtime_state["phase"] = f"Swiss Round {round_idx + 1}"
                    runtime_state["current_round"] = str(round_idx + 1)
                    feed_messages.append(f"⚔️ [bold blue]Swiss Round {round_idx+1}[/] ({len(active_configs)} survivors)")
                    pause_scene_transition()
                    pairings = (len(active_configs) // 2) + (1 if len(active_configs) % 2 != 0 else 0)
                    phase_total = pairings * swaps_per_pair * dynamic_reps
                    if phase_task_id is None:
                        phase_task_id = phase_progress.add_task(
                            f"[bold white]Swiss {scenario.name} R{round_idx + 1}",
                            total=phase_total,
                        )
                    else:
                        phase_progress.reset(
                            phase_task_id,
                            description=f"[bold white]Swiss {scenario.name} R{round_idx + 1}",
                            total=phase_total,
                        )
                    
                    active_configs.sort(key=lambda c: stats[c.name][2], reverse=True)
                    swiss_tasks: List[Tuple] = []
                    
                    for i in range(0, len(active_configs)-1, 2):
                        c1, c2 = active_configs[i], active_configs[i+1]
                        swaps = (False, True) if tcfg.swap_sides else (False,)
                        for _rep in range(dynamic_reps):
                            for swap in swaps:
                                swiss_tasks.append((c1, c2, scenario, rng.randint(0, 999999), swap))
                                if len(swiss_tasks) >= batch_size:
                                    run_task_batch(swiss_tasks, executor, overall_task_id, scenario_utils, False)
                                    swiss_tasks.clear()
                            
                    if len(active_configs) % 2 != 0:
                        c1, c2 = active_configs[-1], rng.choice(active_configs[:-1])
                        swaps = (False, True) if tcfg.swap_sides else (False,)
                        for _rep in range(dynamic_reps):
                            for swap in swaps:
                                swiss_tasks.append((c1, c2, scenario, rng.randint(0, 999999), swap))
                                if len(swiss_tasks) >= batch_size:
                                    run_task_batch(swiss_tasks, executor, overall_task_id, scenario_utils, False)
                                    swiss_tasks.clear()
                    if swiss_tasks:
                        run_task_batch(swiss_tasks, executor, overall_task_id, scenario_utils, False)
                    phase_index += 1
                    overall_progress.update(overall_task_id, completed=float(phase_index))
                    pause_scene_transition()

        if executor:
            executor.shutdown()

    # -----------------------------
    # Post-Tournament Execution
    # -----------------------------

    raw_file.close()

    if raw_writer["writer"] is None:
        df_raw = pd.DataFrame()
    else:
        df_raw = pd.read_csv(raw_csv_path)

    report_console = Console(record=True)
    make_tui_report(report_console, df_raw, stamp_dir)
    report_console.save_text(os.path.join(stamp_dir, "tournament_report.txt"))
    console.print(Panel.fit(f"Wrote outputs to: {stamp_dir}", title="Done", border_style="bold green"))

if __name__ == "__main__":
    main()