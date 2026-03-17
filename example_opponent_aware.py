# Example usage of opponent-aware negotiation strategies

from agent.Group37_Negotiator import Group37_Negotiator
from components.opponent_model import FrequencyAnalysisModel, BayesianUtilityModel, StrategyModel
from components.bidding import OpponentAwareBidding
from components.acceptance import OpponentAwareAcceptance

# Example 1: Frequency analysis model
frequency_model = FrequencyAnalysisModel()
negotiator_freq = Group37_Negotiator(
    bidding_strategy=OpponentAwareBidding(base_threshold=0.85, opponent_model=frequency_model),
    acceptance_strategy=OpponentAwareAcceptance(base_threshold=0.75, opponent_model=frequency_model),
    opponent_model=frequency_model,
)

# Example 2: Bayesian utility model
bayesian_model = BayesianUtilityModel()
negotiator_bayes = Group37_Negotiator(
    bidding_strategy=OpponentAwareBidding(base_threshold=0.85, opponent_model=bayesian_model),
    acceptance_strategy=OpponentAwareAcceptance(base_threshold=0.75, opponent_model=bayesian_model),
    opponent_model=bayesian_model,
)

# Example 3: Strategy model (linear / gaussian / wavelet)
for strategy_type in ("linear", "gaussian", "wavelet"):
    strategy_model = StrategyModel(strategy_type=strategy_type)
    negotiator = Group37_Negotiator(
        bidding_strategy=OpponentAwareBidding(base_threshold=0.85, opponent_model=strategy_model),
        acceptance_strategy=OpponentAwareAcceptance(base_threshold=0.75, opponent_model=strategy_model),
        opponent_model=strategy_model,
    )
    print(f"Created negotiator using StrategyModel(strategy_type='{strategy_type}')")

print("Opponent-aware negotiators created successfully!")
print("Each negotiator will now adapt its bidding and acceptance based on opponent behavior.")