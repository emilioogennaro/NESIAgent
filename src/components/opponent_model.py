from abc import ABC, abstractmethod
from negmas import Outcome
from negmas.sao import SAOState

class OpponentModel(ABC):
    """Abstract base class for tracking and predicting opponent behavior."""
    
    @abstractmethod
    def update(self, offer: Outcome, state: SAOState):
        """Update internal models based on the opponent's latest offer."""
        pass

class NoOpponentModel(OpponentModel):
    """A dummy model for when we are not actively tracking the opponent."""
    
    def update(self, offer: Outcome, state: SAOState):
        pass # Do nothing