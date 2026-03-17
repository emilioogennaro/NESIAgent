# Running the Tournament with Opponent Modeling

## Quick Start

```bash
# Activate your virtual environment
./venv/Scripts/Activate.ps1

# Run the tournament
python src/evaluation/run_tournament.py
```

## What Happens

The tournament now automatically:

1. **Discovers all strategies** from your components:
   - Acceptance strategies: `StaticThresholdAcceptance`, `AspirationalAcceptance`, `OpponentAwareAcceptance`, etc.
   - Bidding strategies: `RandomAboveThresholdBidding`, `OpponentAwareBidding`, `LinearBidding`, etc.
   - Opponent models: `NoOpponentModel`, `OfferHistoryModel`, `FrequencyOpponentModel`, `ConcessionOpponentModel`, `OpponentTypeClassifier`, `PreferenceEstimationModel`

2. **Creates strategy combinations** - For example:
   - OpponentAwareAcceptance + OpponentAwareBidding + FrequencyOpponentModel
   - OpponentAwareAcceptance + OpponentAwareBidding + ConcessionOpponentModel
   - AspirationalAcceptance + LinearBidding + OpponentTypeClassifier
   - And many more...

3. **Runs a full tournament** where all agent configurations play against each other

4. **Shares opponent models** - The bidding and acceptance strategies now receive the same opponent model instance, allowing them to query the opponent's behavior for intelligent adaptation

## How Opponent Modeling Works in the Tournament

When each negotiation happens:

```
1. Your agent receives an offer from opponent
   ↓
2. OpponentModel.update() tracks opponent behavior
   ↓
3. Bidding strategy queries opponent model:
   - opponent_type (hardliner/conceder/moderate)
   - concession_rate (how quickly they concede)
   - offer_frequency (what they tend to propose)
   ↓
4. Bidding strategy adapts its threshold based on opponent type
   ↓
5. Acceptance strategy also queries opponent model:
   - Same insights to decide whether to accept
   ↓
6. Generate counter-offer or accept strategically
```

## Output

The tournament will create a `tournament_out/` directory with:
- `results.csv` - Detailed match results and statistics
- `agents_summary.txt` - Summary of all agent configurations tested
- `timestamp/` - Results timestamped for comparison

## Key Strategy Combinations to Watch

### Against Hardliners
```python
OpponentAwareAcceptance + OpponentAwareBidding + ConcessionOpponentModel
# More lenient with acceptance, slower to concede
```

### Against Conceders
```python
OpponentAwareAcceptance + OpponentAwareBidding + ConcessionOpponentModel
# More demanding, willing to concede faster
```

### Balanced Approach
```python
OpponentAwareAcceptance + OpponentAwareBidding + OpponentTypeClassifier
# Classifies opponent type dynamically and adapts
```

### Value-Based Negotiation
```python
OpponentAwareAcceptance + OpponentAwareBidding + PreferenceEstimationModel
# Learns what the opponent values and exploits it
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

## Analyzing Results

After the tournament, check `tournament_out/results.csv`:

```
agentA, agentB, scenario, agentA_utility, agentB_utility, ...
OpponentAwareBidding(...), LinearBidding(...), negotiation, 0.85, 0.72, ...
```

Best performers were likely using opponent modeling to adapt to diverse strategies!
