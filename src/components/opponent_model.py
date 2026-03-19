# type: ignore
from abc import ABC, abstractmethod
from collections import Counter
import math
from typing import Any, Dict, List, Optional

from negmas import Outcome
from negmas.sao import SAOState


class OpponentModel(ABC):
    """Abstract base class for tracking and predicting opponent behavior."""

    @abstractmethod
    def update(self, offer: Any, state: Any):
        """Update internal models based on the opponent's latest offer."""
        pass


class NoOpponentModel(OpponentModel):
    """A dummy model for when we are not actively tracking the opponent."""

    def update(self, offer: Any, state: Any):
        pass


class FrequencyAnalysisModel(OpponentModel):
    """Tracks how often specific outcomes are offered by the opponent."""

    def __init__(self):
        self.offer_counts: Counter[Outcome] = Counter()
        self.total_offers = 0

    def update(self, offer: Outcome, state: SAOState):
        if offer is not None:
            self.offer_counts[offer] += 1
            self.total_offers += 1

    def get_most_frequent_offer(self) -> Optional[Outcome]:
        """Return the opponent's most frequently offered outcome."""
        if not self.offer_counts:
            return None
        return self.offer_counts.most_common(1)[0][0]

    def get_offer_frequency(self, offer: Outcome) -> float:
        """Return how often an outcome is offered (0.0..1.0)."""
        if self.total_offers == 0:
            return 0.0
        return self.offer_counts[offer] / self.total_offers


class BayesianUtilityModel(OpponentModel):
    """Estimates opponent utility using a simple Bayesian frequency model."""

    def __init__(self):
        self.value_counts: Dict[int, Counter] = {}
        self.issue_counts: Dict[int, int] = {}

    def update(self, offer: Outcome, state: SAOState):
        if offer is None or not isinstance(offer, tuple):
            return

        for i, v in enumerate(offer):
            if i not in self.value_counts:
                self.value_counts[i] = Counter()
                self.issue_counts[i] = 0
            self.value_counts[i][v] += 1
            self.issue_counts[i] += 1

    def estimate_utility(self, offer: Outcome) -> float:
        """Estimate opponent utility (0..1) based on value frequencies."""
        if offer is None or not isinstance(offer, tuple) or not self.issue_counts:
            return 0.5

        total_score = 0.0
        for i, v in enumerate(offer):
            if i not in self.value_counts or self.issue_counts[i] == 0:
                total_score += 0.5
                continue

            count = self.value_counts[i].get(v, 0)
            score = (count + 1) / (self.issue_counts[i] + len(self.value_counts[i]))
            total_score += score

        return total_score / len(offer)


class StrategyModel(OpponentModel):
    """Models opponent strategy using utility progression over time.

    Supported strategies:
      - linear: linear extrapolation
      - gaussian: RBF-weighted prediction
      - wavelet: simple smoothing
    """

    def __init__(self, strategy_type: str = "linear"):
        self.strategy_type = strategy_type
        self.times: List[float] = []
        self.utilities: List[Outcome] = []

    def update(self, offer: Outcome, state: SAOState):
        if offer is None:
            return

        self.times.append(state.relative_time)
        self.utilities.append(offer)

    def _offer_utility(self, offer: Outcome, ufun: Any) -> float:
        try:
            return float(ufun(offer))
        except Exception:
            return 0.5

    def predict_next_utility(self, ufun: Any, next_time: float) -> float:
        """Predict opponent utility at a future relative time."""
        if not self.times or not self.utilities:
            return 0.5

        values = [self._offer_utility(o, ufun) for o in self.utilities]
        times = list(self.times)

        if self.strategy_type == "gaussian":
            return self._predict_gaussian(times, values, next_time)
        if self.strategy_type == "wavelet":
            return self._predict_wavelet(values)
        return self._predict_linear(times, values, next_time)

    def _predict_linear(self, times: List[float], values: List[float], next_time: float) -> float:
        if len(times) < 2:
            return values[-1]

        n = len(times)
        x_mean = sum(times) / n
        y_mean = sum(values) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(times, values))
        den = sum((x - x_mean) ** 2 for x in times)
        if den == 0:
            return values[-1]
        slope = num / den
        intercept = y_mean - slope * x_mean
        return max(0.0, min(1.0, intercept + slope * next_time))

    def _rbf_kernel(self, x: float, y: float, length_scale: float = 0.2) -> float:
        d = x - y
        return math.exp(-0.5 * (d / length_scale) ** 2)

    def _predict_gaussian(self, times: List[float], values: List[float], next_time: float) -> float:
        weights = [self._rbf_kernel(next_time, t) for t in times]
        total = sum(w * v for w, v in zip(weights, values))
        denom = sum(weights)
        if denom == 0:
            return values[-1]
        return max(0.0, min(1.0, total / denom))

    def _predict_wavelet(self, values: List[float]) -> float:
        if len(values) < 3:
            return values[-1]
        smoothed = sum(values[-3:]) / 3.0
        trend = values[-1] - values[-2]
        return max(0.0, min(1.0, smoothed + trend * 0.5))