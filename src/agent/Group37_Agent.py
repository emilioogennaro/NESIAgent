# type: ignore
from abc import ABC, abstractmethod
from collections import Counter
import math
from typing import Any, Dict, List, Optional

from negmas import Outcome
from negmas.preferences import UtilityFunction
from negmas.sao import SAONegotiator, SAOResponse, ResponseType, SAOState


# Opponent model classes

class OpponentModel(ABC):
    """Abstract base class for tracking and predicting opponent behavior."""

    @abstractmethod
    def update(self, offer: Any, state: Any, ufun: Any = None):
        pass

    def get_opponent_type(self) -> str:
        return "unknown"

    def get_concession_rate(self) -> float:
        return 0.0

    def get_offer_frequency(self, offer: Any) -> float:
        return 0.0

    def estimate_utility(self, offer: Any) -> float:
        return 0.5

    def predict_next_utility(self, ufun: Any, next_time: float) -> float:
        return 0.5

    def predict_concession_point(self, current_time: float) -> float:
        return 1.0


class FrequencyAnalysisModel(OpponentModel):
    """Tracks how often specific outcomes are offered by the opponent."""

    def __init__(self):
        self.offer_counts: Counter[Outcome] = Counter()
        self.total_offers = 0
        self.offer_history: List[Outcome] = []
        self.utility_history: List[float] = []

    def update(self, offer: Outcome, state: SAOState, ufun: Any = None):
        if offer is not None:
            self.offer_counts[offer] += 1
            self.total_offers += 1
            self.offer_history.append(offer)
            if ufun is not None:
                try:
                    self.utility_history.append(float(ufun(offer)))
                except Exception:
                    pass

    def get_most_frequent_offer(self) -> Optional[Outcome]:
        if not self.offer_counts:
            return None
        return self.offer_counts.most_common(1)[0][0]

    def get_offer_frequency(self, offer: Outcome) -> float:
        if self.total_offers == 0:
            return 0.0
        return self.offer_counts[offer] / self.total_offers

    def estimate_utility(self, offer: Outcome) -> float:
        if offer is None or self.total_offers == 0 or not isinstance(offer, tuple):
            return 0.5

        score = 0.0
        for i, v in enumerate(offer):
            value_freq = sum(c for o, c in self.offer_counts.items() if isinstance(o, tuple) and len(o) > i and o[i] == v)
            score += (value_freq + 1) / (self.total_offers + len(self.offer_counts))

        return max(0.0, min(1.0, score / len(offer)))

    def _compute_slope(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        n = len(values)
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(values) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
        den = sum((x - x_mean) ** 2 for x in xs)
        if den == 0:
            return 0.0
        return num / den

    def get_concession_rate(self) -> float:
        if not self.utility_history:
            return 0.0
        slope = self._compute_slope(self.utility_history)
        return max(0.0, min(1.0, slope / 0.05))

    def get_opponent_type(self) -> str:
        if not self.utility_history:
            return "unknown"
        slope = self._compute_slope(self.utility_history)
        if slope > 0.02:
            return "conceder"
        if slope < -0.02:
            return "hardliner"
        return "balanced"

    def predict_next_utility(self, ufun: Any, next_time: float) -> float:
        if not self.utility_history:
            return 0.5
        if len(self.utility_history) == 1:
            return self.utility_history[-1]
        slope = self._compute_slope(self.utility_history)
        return max(0.0, min(1.0, self.utility_history[-1] + slope))

    def predict_concession_point(self, current_time: float) -> float:
        rate = self.get_concession_rate()
        if rate <= 0:
            return 1.0
        return max(0.0, min(1.0, current_time + (1.0 - current_time) * (1.0 - rate)))


# Bidding strategy classes

class BiddingStrategy(ABC):
    def __init__(self, opponent_model: Optional[OpponentModel] = None):
        self.opponent_model = opponent_model

    @abstractmethod
    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Optional[Outcome]:
        pass

    def _find_outcome_for_utility(self, target_utility: float, ufun: UtilityFunction, nmi: Any, samples: int = 200) -> Outcome:
        best_candidate = None
        smallest_diff = float('inf')
        for _ in range(samples):
            candidate = nmi.random_outcome()
            if candidate is None:
                continue
            util = float(ufun(candidate))
            diff = abs(util - target_utility)
            if diff < smallest_diff:
                smallest_diff = diff
                best_candidate = candidate
            if smallest_diff < 0.005:
                break
        return best_candidate if best_candidate is not None else ufun.extreme_outcomes()[1]


class OpponentAwareBidding(BiddingStrategy):
    def __init__(self, base_threshold: float = 0.9, opponent_model: Optional[OpponentModel] = None):
        super().__init__(opponent_model)
        self.base_threshold = base_threshold

    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Optional[Outcome]:
        if state.step == 0:
            return ufun.extreme_outcomes()[1]

        threshold = self.base_threshold

        if self.opponent_model:
            if hasattr(self.opponent_model, 'get_opponent_type'):
                opponent_type = self.opponent_model.get_opponent_type()
                if opponent_type == "hardliner":
                    threshold += 0.1
                elif opponent_type == "conceder":
                    threshold -= 0.1

            if hasattr(self.opponent_model, 'get_concession_rate'):
                concession_rate = self.opponent_model.get_concession_rate()
                if concession_rate < 0.3:
                    threshold += 0.05

            if hasattr(self.opponent_model, 'get_offer_frequency') and state.current_offer is not None:
                frequency = self.opponent_model.get_offer_frequency(state.current_offer)
                if frequency > 0.4:
                    threshold += 0.05

            if hasattr(self.opponent_model, 'estimate_utility') and state.current_offer is not None:
                est = self.opponent_model.estimate_utility(state.current_offer)
                if est > 0.8:
                    threshold += 0.05
                elif est < 0.3:
                    threshold -= 0.05

            if hasattr(self.opponent_model, 'predict_next_utility'):
                next_time = min(1.0, (state.relative_time or 0.0) + 0.05)
                predicted = self.opponent_model.predict_next_utility(ufun, next_time)
                if predicted > 0.8:
                    threshold += 0.05
                elif predicted < 0.3:
                    threshold -= 0.05

            if hasattr(self.opponent_model, 'predict_concession_point'):
                next_concession = self.opponent_model.predict_concession_point(state.relative_time or 0.0)
                if next_concession < 0.8:
                    threshold -= 0.05

        threshold = max(float(ufun.reserved_value), min(0.98, threshold))

        target_threshold = max(float(ufun.reserved_value), threshold)

        for _ in range(1000):
            candidate = nmi.random_outcome()
            if candidate is None:
                continue
            if float(ufun(candidate)) >= target_threshold:
                return candidate

        return ufun.extreme_outcomes()[1]


# Acceptance strategy classes

class AcceptanceStrategy(ABC):
    def __init__(self, opponent_model: Optional[OpponentModel] = None):
        self.opponent_model = opponent_model

    @abstractmethod
    def evaluate(self, offer: Any, state: Any, ufun: Any) -> bool:
        pass


class HybridAcceptance(AcceptanceStrategy):
    def __init__(self, aspiration_weight: float = 0.4, opponent_weight: float = 0.3, time_weight: float = 0.3, opponent_model: Optional[OpponentModel] = None):
        super().__init__(opponent_model=opponent_model)
        self.aspiration_weight = aspiration_weight
        self.opponent_weight = opponent_weight
        self.time_weight = time_weight
        self.opponent_utilities: List[float] = []

    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        if offer is None:
            return False

        offer_utility = float(ufun(offer))
        time_progress = state.relative_time if state.relative_time is not None else 0

        aspiration_level = 0.3 + 0.7 * ((1 - time_progress) ** 1.5)
        aspiration_score = min(1.0, offer_utility / aspiration_level) if aspiration_level > 0 else 0

        self.opponent_utilities.append(offer_utility)
        if len(self.opponent_utilities) > 1:
            avg_opponent_utility = sum(self.opponent_utilities) / len(self.opponent_utilities)
            opponent_score = min(1.0, avg_opponent_utility)
        else:
            opponent_score = 0.5

        time_score = 0.3 + 0.7 * time_progress

        decision_score = (
            self.aspiration_weight * aspiration_score +
            self.opponent_weight * opponent_score +
            self.time_weight * time_score
        )

        return decision_score >= 0.5


# Group37 agent wrapper

class Group37Agent(SAONegotiator):
    def __init__(self, *args, bidding_strategy: Optional[BiddingStrategy] = None, acceptance_strategy: Optional[AcceptanceStrategy] = None, opponent_model: Optional[OpponentModel] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.opponent_model = opponent_model or FrequencyAnalysisModel()
        self.bidding_strategy = bidding_strategy or OpponentAwareBidding(opponent_model=self.opponent_model)
        self.acceptance_strategy = acceptance_strategy or HybridAcceptance(opponent_model=self.opponent_model)

    def __call__(self, state: SAOState, *args: Any, **kwargs: Any) -> SAOResponse:
        offer = state.current_offer
        if offer is not None:
            self.opponent_model.update(offer, state, self.ufun)
            if self.acceptance_strategy.evaluate(offer, state, self.ufun):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        my_proposal = self.bidding_strategy.generate(state, self.ufun, self.nmi)
        return SAOResponse(ResponseType.REJECT_OFFER, my_proposal)


def build_group37_agent(name: str = "Group37Agent") -> Group37Agent:
    opp = FrequencyAnalysisModel()
    acc = HybridAcceptance(opponent_model=opp)
    bid = OpponentAwareBidding(opponent_model=opp)
    return Group37Agent(name=name, acceptance_strategy=acc, bidding_strategy=bid, opponent_model=opp)
