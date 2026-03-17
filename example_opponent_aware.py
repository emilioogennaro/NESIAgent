# Example usage of opponent-aware negotiation strategies

from agent.Group37_Negotiator import Group37_Negotiator
from components.opponent_model import (
    OfferHistoryModel,
    FrequencyOpponentModel,
    ConcessionOpponentModel,
    PreferenceEstimationModel,
    OpponentTypeClassifier
)
from components.bidding import OpponentAwareBidding
from components.acceptance import OpponentAwareAcceptance

# Example 1: Using opponent-aware bidding and acceptance with frequency modeling
frequency_model = FrequencyOpponentModel()
negotiator1 = Group37_Negotiator(
    bidding_strategy=OpponentAwareBidding(base_threshold=0.85, opponent_model=frequency_model),
    acceptance_strategy=OpponentAwareAcceptance(base_threshold=0.75, opponent_model=frequency_model),
    opponent_model=frequency_model
)

# Example 2: Using concession modeling for more adaptive behavior
concession_model = ConcessionOpponentModel()
negotiator2 = Group37_Negotiator(
    bidding_strategy=OpponentAwareBidding(base_threshold=0.9, opponent_model=concession_model),
    acceptance_strategy=OpponentAwareAcceptance(base_threshold=0.8, opponent_model=concession_model),
    opponent_model=concession_model
)

# Example 3: Using opponent type classification
type_model = OpponentTypeClassifier()
negotiator3 = Group37_Negotiator(
    bidding_strategy=OpponentAwareBidding(base_threshold=0.88, opponent_model=type_model),
    acceptance_strategy=OpponentAwareAcceptance(base_threshold=0.78, opponent_model=type_model),
    opponent_model=type_model
)

# Example 4: Using preference estimation for value-based adaptation
preference_model = PreferenceEstimationModel()
negotiator4 = Group37_Negotiator(
    bidding_strategy=OpponentAwareBidding(base_threshold=0.85, opponent_model=preference_model),
    acceptance_strategy=OpponentAwareAcceptance(base_threshold=0.75, opponent_model=preference_model),
    opponent_model=preference_model
)

print("Opponent-aware negotiators created successfully!")
print("Each negotiator will now adapt its bidding and acceptance based on opponent behavior.")