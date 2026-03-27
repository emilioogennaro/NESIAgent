# NESI Negotiation Agent

This repository implements **NESI**, a negotiation agent developed for CSE3210
Collaborative Artificial Intelegence course.  NESI is built on top of the
[NegMAS](https://github.com/negmas-org/negmas) multi‑agent negotiation framework.

## Repository Layout

- `src/nesi_agent`: Python package containing the NESI agent source code.
- `pyproject.toml`: Project manifest with metadata and dependency declarations.

## Setup & Installation

```bash
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -e .
```

After activation the `negmas` dependency will be available and the package can be imported:

```python
from nesi_agent.agent import NESIAgent
```

## Quick Start of the Tournament

```bash
# Activate your virtual environment
./venv/Scripts/Activate.ps1

# Run the cross-validation tournament
python src/evaluation/run_tournament.py

# Run the evaluation tournament
python src/evaluation/run_tournament_evaluation.py
```

## Customizing the Tournament

Edit the bottom of `run_tournament.py` to modify:

- `ScenarioConfig`: How many issues, values per issue, time steps
- `TournamentConfig`: Number of repetitions per matchup, include self-play
- `reps_per_pair`: How many times each agent pair plays (default 3)
- `n_steps`: How many negotiation rounds per session (default 100)

Example from within `run_tournament.py`:

```python
if __name__ == "__main__":
    scenario_config = ScenarioConfig(
        name="negotiation",
        n_issues=3,           # More complex negotiations
        n_values=50,
        n_steps=100,
        reserved_min=0.0,
        reserved_max=0.2
    )
    
    tournament_config = TournamentConfig(
        out_dir="tournament_out",
        reps_per_pair=5,      # More repetitions for statistical significance
        swap_sides=True,      # Test both agent positions
        include_self_play=False
    )
    
    run_tournament(scenario_config, tournament_config)
```

## Command-Line Flags

Run tournaments with advanced options by passing flags to the command:

```bash
python src/evaluation/run_tournament.py --reps 5 --workers 8 --seed 1234
```

### Output & Control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--out` | string | `tournament_out` | Output directory for results and metadata |
| `--seed` | int | `2026` | Base random number generator seed for reproducibility |

### Tournament Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--reps` | int | `3` | Static mode: repetitions per agent pair per scenario |
| `--no-swap` | flag | - | Disable swapping utility assignments (test agents only in one role) |
| `--self-play` | flag | - | Include agents playing against themselves |
| `--max-pairs` | int | `None` | Cap the maximum number of pairings (useful for testing) |
| `--timeout` | int | `30` | Limits the duration of a match up to a particular number of seconds. |

### Scenario Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--s-large` | flag | - | Enable larger scenario: 5 issues, 80 values per issue, 160 negotiation steps |

### Performance & Execution

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--speed` | float | `1.0` | UI simulation speed multiplier. `0.05` = slow; `>2.0` = multiprocessing mode |
| `--workers` | int | `4` | Maximum number of worker processes (multiprocessing mode only) |
| `--rss-limit-gb` | int | `None` | Hard memory cap in GB (stops tournament if exceeded) |
| `--scene-pause` | float | `0.8` | Pause duration in seconds between UI scene transitions |

### Dynamic Tournament Mode

Enable advanced adaptive scheduling with `--dynamic`. The tournament prunes underperforming agents and focuses on promising matchups:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dynamic` | flag | - | Enable Swiss-Bandit dynamic tournament scheduling |
| `--grace-matches` | int | `5` | Minimum matches per agent before pruning (Dynamic only) |
| `--swiss-rounds` | int | `5` | Number of adaptive Swiss rounds per scenario (Dynamic only) |
| `--dynamic-reps` | int | `1` | Repetitions per dynamic pairing (Dynamic only) |
| `--c-value` | float | `1.96` | Confidence interval multiplier for pruning (Dynamic only) |

### Usage Examples

**Standard tournament with 5 repetitions:**
```bash
python src/evaluation/run_tournament.py --reps 5
```

**Fast parallel execution with 8 workers:**
```bash
python src/evaluation/run_tournament.py --speed 10.0 --workers 8
```

**Reproducible results with fixed seed:**
```bash
python src/evaluation/run_tournament.py --seed 42 --reps 10
```

**Testing on larger scenario with agent self-play:**
```bash
python src/evaluation/run_tournament.py --s-large --self-play
```

**Dynamic adaptive tournament with memory limit:**
```bash
python src/evaluation/run_tournament.py --dynamic --grace-matches 3 --swiss-rounds 7 --rss-limit-gb 16
```

## Output

The tournament creates a `tournament_out/` directory with:
- `results.csv` - Detailed match results and statistics
- `agents_summary.txt` - Summary of all agent configurations tested
- `timestamp/` - Results timestamped for comparison


## Visualisation

For the sake of better displaying and analyzing the results of a tournament run, we implmented a command-line visualisation, containing essential information.

The visualisation contains the following components:
   - Two progress bars - one for the overall completion of the tournament and one for the current scenario;
   - Live ranking of the 10 best agents in the tournament, given their mean utility + agreement rate;
   - Live match feed, which shows the last executed matches between agents, the final status of the negotiation and the two utilities of the agents.
   - Live stats showing more information about the progress of the current scenario.