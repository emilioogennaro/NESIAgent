import os
# type: ignore
import sys
import json
import time
import math
import random
import inspect
import itertools
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

# Ensure src/ is importable (same idea as your current script)
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
    # Try with filtered kwargs, then fall back to no-arg.
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
            if issubclass(obj, base_cls):
                out.append(obj)
        except TypeError:
            continue
    # deterministic ordering
    out.sort(key=lambda c: (c.__module__, c.__name__))
    return out


# -----------------------------
# Strategy discovery
# -----------------------------

def discover_strategies() -> Tuple[List[StrategySpec], List[StrategySpec], List[StrategySpec]]:
    # Import the concrete modules to search for subclasses.
    import components.acceptance as acc_mod
    import components.bidding as bid_mod
    import components.opponent_model as opp_mod

    acc_classes = _discover_subclasses(acc_mod, AcceptanceStrategy)
    bid_classes = _discover_subclasses(bid_mod, BiddingStrategy)
    opp_classes = _discover_subclasses(opp_mod, OpponentModel)

    # Default parameter grids (extend as you like)
    acceptance_specs: List[StrategySpec] = []
    for c in acc_classes:
        # Example: try a few thresholds if the class supports it
        if "threshold" in (inspect.signature(c).parameters if _has_signature(c) else {}):
            for thr in (0.6, 0.7, 0.8, 0.9):
                acceptance_specs.append(StrategySpec("acceptance", c.__name__, c.__module__, {"threshold": thr}))
        else:
            acceptance_specs.append(StrategySpec("acceptance", c.__name__, c.__module__, {}))

    bidding_specs: List[StrategySpec] = []
    for c in bid_classes:
        params = {}
        # Some of your bidding strategies have optional knobs; sample a small grid
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
        opponent_specs.append(StrategySpec("opponent_model", c.__name__, c.__module__, {}))

    # If you want smaller tournaments, you can cap here:
    # bidding_specs = bidding_specs[:10]

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
    # LUFun.random accepts reserved_value as a number or range (your current code uses a tuple)
    rmin = min(s.reserved_min, s.reserved_max)
    rmax = max(s.reserved_min, s.reserved_max)
    # Add tiny seed noise so A/B ufuns differ even with same scenario seed
    # Cast `issues` to a non-variant typed list to satisfy static checkers
    ufun_a = LUFun.random(issues=cast(List[Any], issues), reserved_value=(rmin, rmax))
    ufun_b = LUFun.random(issues=cast(List[Any], issues), reserved_value=(rmin, rmax))
    return issues, ufun_a, ufun_b


def build_agent(cfg: AgentConfig, name: str):
    # dynamic import by module path stored in StrategySpec
    acc_cls = getattr(__import__(cfg.acceptance.module, fromlist=[cfg.acceptance.cls_name]), cfg.acceptance.cls_name)
    bid_cls = getattr(__import__(cfg.bidding.module, fromlist=[cfg.bidding.cls_name]), cfg.bidding.cls_name)
    opp_cls = getattr(__import__(cfg.opponent_model.module, fromlist=[cfg.opponent_model.cls_name]), cfg.opponent_model.cls_name)

    acc = _instantiate(acc_cls, cfg.acceptance.params)
    bid = _instantiate(bid_cls, cfg.bidding.params)
    opp = _instantiate(opp_cls, cfg.opponent_model.params)

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

    # Swap only utility assignments to mitigate "who got which preference draw"
    ufun_a, ufun_b = (ufun2, ufun1) if swap_ufuns else (ufun1, ufun2)

    mech = SAOMechanism(issues=issues, n_steps=scenario.n_steps)

    agent_a = build_agent(cfg_a, name="AgentA")
    agent_b = build_agent(cfg_b, name="AgentB")

    mech.add(agent_a, ufun=ufun_a)
    mech.add(agent_b, ufun=ufun_b)

    state = mech.run()
    agreement = state.agreement

    # Ensure we convert potential numpy/pandas scalar results to Python floats
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


def run_duel(
    cfg_a: AgentConfig,
    cfg_b: AgentConfig,
    scenario: ScenarioConfig,
    base_seed: int,
    reps: int,
    swap_sides: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in range(reps):
        seed = base_seed + 100000 * (hash(scenario.name) % 1000) + 1000 * r
        rows.append(run_one_session(cfg_a, cfg_b, scenario, seed=seed, swap_ufuns=False))
        if swap_sides:
            rows.append(run_one_session(cfg_a, cfg_b, scenario, seed=seed + 1, swap_ufuns=True))
    return rows


# -----------------------------
# Reporting
# -----------------------------

def make_tui_report(console: Console, df_raw: pd.DataFrame, out_dir: str) -> None:
    console.print(Panel.fit(f"[bold]Tournament finished[/bold]\nRows: {len(df_raw)}\nOut: {out_dir}", title="Status"))

    if df_raw.empty:
        console.print("[red]No results produced.[/red]")
        return

    # Pair-level summary
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

    # Agent-level summary (pool role A and role B)
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

    # Print top agents per scenario
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
            t.add_row(
                str(i),
                str(row.agent),
                f"{agree:0.2f} {_bar(agree)}",
                f"{mu:0.3f}",
                f"{ms:0.1f}",
            )
        console.print(t)

    # Overall summary across scenarios
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
        t2.add_row(
            str(i),
            str(row.agent),
            f"{agree:0.2f} {_bar(agree)}",
            f"{meanu:0.3f}",
            f"{means:0.1f}",
        )
    console.print(t2)

    # Save aggregated outputs
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
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on number of pairs (debugging)")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed")
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

    configs: List[AgentConfig] = [
        AgentConfig(a, b, o)
        for a in acc_specs
        for b in bid_specs
        for o in opp_specs
    ]

    # Build pairs
    if tcfg.include_self_play:
        pairs = list(itertools.combinations_with_replacement(configs, 2))
    else:
        pairs = list(itertools.combinations(configs, 2))

    if tcfg.max_pairs is not None:
        pairs = pairs[: max(0, tcfg.max_pairs)]

    # Save tournament config + strategy inventories
    meta = {
        "tournament_config": asdict(tcfg),
        "scenarios": [asdict(s) for s in scenarios],
        "n_acceptance_specs": len(acc_specs),
        "n_bidding_specs": len(bid_specs),
        "n_opponent_specs": len(opp_specs),
        "n_agent_configs": len(configs),
        "n_pairs": len(pairs),
        "acceptance_specs": [asdict(s) for s in acc_specs],
        "bidding_specs": [asdict(s) for s in bid_specs],
        "opponent_specs": [asdict(s) for s in opp_specs],
    }
    with open(os.path.join(stamp_dir, "tournament_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    console.print(Panel.fit(
        f"Configs: {len(configs)}\nPairs: {len(pairs)}\nScenarios: {len(scenarios)}\nReps/pair: {tcfg.reps_per_pair}\nSwap sides: {tcfg.swap_sides}",
        title="Tournament plan"
    ))

    rows: List[Dict[str, Any]] = []
    total_jobs = len(scenarios) * len(pairs)

    job_iter = (
        (scenario, i, cfg_a, cfg_b)
        for scenario in scenarios
        for i, (cfg_a, cfg_b) in enumerate(pairs)
    )

    for scenario, i, cfg_a, cfg_b in track(job_iter, total=total_jobs, description="Running matches"):
        duel_seed = tcfg.base_seed + 1000000 * (hash(scenario.name) % 1000) + i * 37
        for r in range(tcfg.reps_per_pair):
            seed = duel_seed + 1000 * r
            rows.append(run_one_session(cfg_a, cfg_b, scenario, seed=seed, swap_ufuns=False))
            if tcfg.swap_sides:
                rows.append(run_one_session(cfg_a, cfg_b, scenario, seed=seed + 1, swap_ufuns=True))

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(os.path.join(stamp_dir, "raw_results.csv"), index=False)

    make_tui_report(console, df_raw, stamp_dir)

    # Save the same TUI output to file
    console.save_text(os.path.join(stamp_dir, "tournament_report.txt"))

    console.print(Panel.fit(f"Wrote outputs to: {stamp_dir}", title="Done"))


if __name__ == "__main__":
    main()