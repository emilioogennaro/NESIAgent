# type: ignore
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from collections import defaultdict, Counter
from negmas import Outcome
from negmas.sao import SAOState
import math

class OpponentModel(ABC):
    """Abstract base class for tracking and predicting opponent behavior."""
    
    @abstractmethod
    def update(self, offer: Any, state: Any):
        """Update internal models based on the opponent's latest offer."""
        pass

class NoOpponentModel(OpponentModel):
    """A dummy model for when we are not actively tracking the opponent."""
    
    def update(self, offer: Any, state: Any):
        pass  # Do nothing

class OfferHistoryModel(OpponentModel):
    """Tracks the history of opponent's offers for pattern analysis."""
    
    def __init__(self):
        self.offers: List[Outcome] = []
        self.times: List[float] = []  # relative time when offer was made
    
    def update(self, offer: Outcome, state: SAOState):
        if offer is not None:
            self.offers.append(offer)
            self.times.append(state.relative_time)
    
    def get_offer_history(self) -> List[Outcome]:
        """Return the list of past offers."""
        return self.offers.copy()
    
    def get_recent_offers(self, n: int = 5) -> List[Outcome]:
        """Return the last n offers."""
        return self.offers[-n:] if len(self.offers) >= n else self.offers.copy()

class FrequencyOpponentModel(OpponentModel):
    """Tracks the frequency of different offers to identify preferred outcomes."""
    
    def __init__(self):
        self.offer_counts: Counter[Outcome] = Counter()
        self.total_offers = 0
    
    def update(self, offer: Outcome, state: SAOState):
        if offer is not None:
            self.offer_counts[offer] += 1
            self.total_offers += 1
    
    def get_most_frequent_offer(self) -> Optional[Outcome]:
        """Return the most frequently offered outcome."""
        if not self.offer_counts:
            return None
        return self.offer_counts.most_common(1)[0][0]
    
    def get_offer_frequency(self, offer: Outcome) -> float:
        """Return the frequency of a specific offer (0.0 to 1.0)."""
        if self.total_offers == 0:
            return 0.0
        return self.offer_counts[offer] / self.total_offers
    
    def get_frequent_offers(self, threshold: float = 0.1) -> List[Outcome]:
        """Return offers that appear more than threshold fraction of the time."""
        if self.total_offers == 0:
            return []
        return [offer for offer, count in self.offer_counts.items() 
                if count / self.total_offers >= threshold]

class ConcessionOpponentModel(OpponentModel):
    """Models the opponent's concession behavior over time."""
    
    def __init__(self):
        self.offers: List[Outcome] = []
        self.times: List[float] = []
        # Assume offers are in decreasing utility order for opponent
        # First offer is best, later offers show concession
    
    def update(self, offer: Outcome, state: SAOState):
        if offer is not None:
            self.offers.append(offer)
            self.times.append(state.relative_time)
    
    def get_concession_rate(self) -> float:
        """Estimate concession rate based on offer sequence.
        
        Returns a value between 0 and 1, where:
        - 0: No concession (hardliner)
        - 1: Immediate concession to worst offer
        """
        if len(self.offers) < 2:
            return 0.0
        
        # Simple heuristic: assume first offer is best, last is current concession
        # Rate is how far into negotiation we are when making significant concessions
        if len(self.offers) == 1:
            return 0.0
        
        # If opponent made many offers quickly, they concede fast
        time_span = self.times[-1] - self.times[0] if self.times[0] < self.times[-1] else 1.0
        offer_rate = (len(self.offers) - 1) / time_span if time_span > 0 else 1.0
        
        # Normalize to 0-1 scale (arbitrary scaling)
        return min(1.0, offer_rate / 10.0)  # Assume 10 offers per time unit is fast conceder
    
    def is_conceder(self) -> bool:
        """Classify opponent as a conceder (True) or hardliner (False)."""
        return self.get_concession_rate() > 0.5
    
    def predict_concession_point(self, current_time: float) -> float:
        """Predict at what time the opponent will make their next concession."""
        if len(self.times) < 2:
            return current_time + 0.1  # Default prediction
        
        # Simple linear extrapolation
        if len(self.times) >= 2:
            time_diffs = [self.times[i+1] - self.times[i] for i in range(len(self.times)-1)]
            avg_diff = sum(time_diffs) / len(time_diffs)
            return current_time + avg_diff
        return current_time + 0.1

class PreferenceEstimationModel(OpponentModel):
    """Attempts to estimate opponent's preferences based on offered outcomes."""
    
    def __init__(self):
        self.offers: List[Outcome] = []
        self.issue_weights: Dict[int, float] = {}  # For vector outcomes
        self.value_preferences: Dict[Any, float] = {}  # For categorical values
    
    def update(self, offer: Outcome, state: SAOState):
        if offer is not None:
            self.offers.append(offer)
            self._update_preferences(offer)
    
    def _update_preferences(self, offer: Outcome):
        """Update preference estimates based on new offer."""
        if isinstance(offer, tuple):
            # Assume tuple represents issue values
            for i, value in enumerate(offer):
                if i not in self.issue_weights:
                    self.issue_weights[i] = 1.0  # Initialize equally
                # Simple frequency-based preference
                if value not in self.value_preferences:
                    self.value_preferences[value] = 0.0
                self.value_preferences[value] += 1.0 / len(self.offers)
    
    def get_preferred_values(self, issue_index: int = 0) -> List[Any]:
        """Return values for an issue sorted by estimated preference."""
        if not isinstance(self.offers[0], tuple) if self.offers else True:
            return []
        
        # Get all values for this issue
        values = [offer[issue_index] for offer in self.offers if len(offer) > issue_index]
        if not values:
            return []
        
        # Sort by frequency (simple preference estimate)
        value_counts = Counter(values)
        return [value for value, _ in value_counts.most_common()]
    
    def estimate_utility(self, offer: Outcome) -> float:
        """Estimate opponent's utility for an offer (0.0 to 1.0)."""
        if not self.offers:
            return 0.5  # Neutral
        
        if isinstance(offer, tuple) and self.offers:
            # Simple additive utility based on value preferences
            total_score = 0.0
            for i, value in enumerate(offer):
                if value in self.value_preferences:
                    total_score += self.value_preferences[value]
                else:
                    total_score += 0.5  # Neutral for unseen values
            
            # Normalize by number of issues
            num_issues = len(offer)
            return total_score / num_issues if num_issues > 0 else 0.5
        
        return 0.5  # Default

class OpponentTypeClassifier(OpponentModel):
    """Classifies opponent behavior into common negotiation types."""
    
    def __init__(self):
        self.offers: List[Outcome] = []
        self.times: List[float] = []
        self.opponent_type: str = "unknown"
    
    def update(self, offer: Outcome, state: SAOState):
        if offer is not None:
            self.offers.append(offer)
            self.times.append(state.relative_time)
            self._classify_opponent()
    
    def _classify_opponent(self):
        """Classify opponent based on offer patterns."""
        if len(self.offers) < 3:
            self.opponent_type = "unknown"
            return
        
        # Simple heuristics
        time_span = self.times[-1] - self.times[0]
        num_offers = len(self.offers)
        
        if time_span < 0.3 and num_offers > 5:
            self.opponent_type = "conceder"  # Quick concessions
        elif time_span > 0.7 and num_offers <= 3:
            self.opponent_type = "hardliner"  # Slow to concede
        elif num_offers > 10:
            self.opponent_type = "random"  # Many offers, possibly random
        else:
            self.opponent_type = "moderate"  # Balanced behavior
    
    def get_opponent_type(self) -> str:
        """Return the classified opponent type."""
        return self.opponent_type
    
    def get_type_confidence(self) -> float:
        """Return confidence in the classification (0.0 to 1.0)."""
        if self.opponent_type == "unknown":
            return 0.0
        # Simple confidence based on number of observations
        return min(1.0, len(self.offers) / 10.0)