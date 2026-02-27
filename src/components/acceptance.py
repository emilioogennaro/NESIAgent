from abc import ABC, abstractmethod
from negmas import Outcome
from negmas.sao import SAOState
from negmas.preferences import UtilityFunction

class AcceptanceStrategy(ABC):
    """Abstract base class for all acceptance strategies."""
    
    @abstractmethod
    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        """Evaluate the offer and return True to accept, False to reject."""
        pass

class StaticThresholdAcceptance(AcceptanceStrategy):
    """A simple strategy that accepts any offer above a fixed utility threshold."""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        # Prevent errors on the very first turn when the offer is None
        if offer is None:
            return False
            
        return ufun(offer) >= self.threshold