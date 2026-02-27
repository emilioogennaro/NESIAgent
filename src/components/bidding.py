from abc import ABC, abstractmethod
from typing import Any # Use Python's built-in Any instead
from negmas import Outcome
from negmas.sao import SAOState
from negmas.preferences import UtilityFunction

class BiddingStrategy(ABC):
    """Abstract base class for all bidding strategies."""
    
    @abstractmethod
    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Outcome | None:
        """Generate and return an Outcome to propose."""
        pass

class RandomAboveThresholdBidding(BiddingStrategy):
    """Generates a random offer that meets a minimum utility requirement."""
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Outcome | None:
        # Always ask for the absolute best on the first turn
        if state.step == 0:
            return ufun.extreme_outcomes()[1]

        # The nmi object still has all its methods at runtime!
        for _ in range(1000):
            candidate = nmi.random_outcome()
            if ufun(candidate) >= self.threshold:
                return candidate
                
        return ufun.extreme_outcomes()[1]