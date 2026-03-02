from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional
from negmas import Outcome
from negmas.sao import SAOState
from negmas.preferences import UtilityFunction

# ---- Base class ----

class BiddingStrategy(ABC):
    """Abstract base class for all bidding strategies."""
    
    @abstractmethod
    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Optional[Outcome]:
        """Generate and return an outcome to propose."""
        pass

    def _find_outcome_for_utility(self, target_utility: float, ufun: UtilityFunction, nmi: Any, samples: int = 200) -> Outcome:
        """Finds an outcome that provides a utility as close to the target as possible.
        
        Safely works with categorical data (strings/discrete values) and respects 
        non-linear utility functions by searching in utility space.
        """
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


# ---- Simple & Randomized strategies ----

class HardlinerBidding(BiddingStrategy):
    """Refuses to concede, always demanding the maximum possible utility."""
    
    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Optional[Outcome]:
        return ufun.extreme_outcomes()[1]


class RandomAboveThresholdBidding(BiddingStrategy):
    """Generates random offers meeting the utility function's reserved value."""
    
    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Optional[Outcome]:
        if state.step == 0:
            return ufun.extreme_outcomes()[1]

        floor_utility = float(ufun.reserved_value)

        for _ in range(1000):
            candidate = nmi.random_outcome()
            if float(ufun(candidate)) >= floor_utility:
                return candidate
                
        return ufun.extreme_outcomes()[1]


# ---- Time-based general class ----

class TimeBasedBiddingStrategy(BiddingStrategy, ABC):
    """Base class for strategies that concede utility over time down to the reserved value."""

    @abstractmethod
    def get_concession_factor(self, progress: float) -> float:
        """Map linear progress [0.0, 1.0] to a curved concession factor [0.0, 1.0]."""
        pass

    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Optional[Outcome]:
        progress = state.relative_time 
        factor = self.get_concession_factor(progress)
        
        max_utility = float(ufun(ufun.extreme_outcomes()[1]))
        floor_utility = float(ufun.reserved_value)
        
        target_utility = max_utility - factor * (max_utility - floor_utility)
        return self._find_outcome_for_utility(target_utility, ufun, nmi)


# ---- Specific time-based strategies ----

class LinearBidding(TimeBasedBiddingStrategy):
    def get_concession_factor(self, progress: float) -> float:
        return progress

class StaircaseBidding(TimeBasedBiddingStrategy):
    def __init__(self, steps: int = 5):
        self.steps = max(1, steps)

    def get_concession_factor(self, progress: float) -> float:
        if progress >= 1.0: return 1.0
        step_width = 1.0 / self.steps
        return math.floor(progress / step_width) * step_width

class HardHeadedButScaredBasedBidding(TimeBasedBiddingStrategy):
    def __init__(self, concession_exponent: float = 2.0):
        self.concession_exponent = concession_exponent

    def get_concession_factor(self, progress: float) -> float:
        return progress ** self.concession_exponent

class LateDropBasedBidding(TimeBasedBiddingStrategy):
    def __init__(self, time_threshold: float = 0.8, collapse_exponent: float = 5.0):
        self.time_threshold = time_threshold
        self.collapse_exponent = collapse_exponent

    def get_concession_factor(self, progress: float) -> float:
        if progress < self.time_threshold: return 0.0
        norm = (progress - self.time_threshold) / (1 - self.time_threshold)
        return norm ** self.collapse_exponent

class NiceButGetsPissedBasedBidding(TimeBasedBiddingStrategy):
    def __init__(self, concession_exponent: float = 0.5):
        self.concession_exponent = concession_exponent

    def get_concession_factor(self, progress: float) -> float:
        return progress ** self.concession_exponent


# ---- Adaptive base classes ----

class AdaptiveBiddingStrategy(BiddingStrategy, ABC):
    """Base class for strategies that react to the opponent's behavior over time."""
    
    def __init__(self):
        self.last_opponent_offer: Optional[Outcome] = None
        self.last_opponent_utility: float = 0.0
        self.current_target_utility: Optional[float] = None
        
    def _update_opponent_history(self, offer: Optional[Outcome], ufun: UtilityFunction):
        """Updates the memory of the opponent's previous offer and its utility for us."""
        if offer is not None:
            self.last_opponent_offer = offer
            self.last_opponent_utility = float(ufun(offer))

    def _initialize_target_utility(self, ufun: UtilityFunction) -> float:
        """Returns the maximum possible utility to initialize the target."""
        if self.current_target_utility is None:
            return float(ufun(ufun.extreme_outcomes()[1]))
        return self.current_target_utility


# ---- Adaptive strategies ----

class TitForTatBidding(AdaptiveBiddingStrategy):
    def __init__(self, strictness: float = 1.0):
        super().__init__()
        self.strictness = strictness 

    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Optional[Outcome]:
        # Pylance now knows target is strictly a float
        target: float = self._initialize_target_utility(ufun)
        offer = state.current_offer

        if offer is None or state.step == 0:
            self.current_target_utility = target
            return ufun.extreme_outcomes()[1]

        current_opponent_util = float(ufun(offer))

        if self.last_opponent_offer is not None:
            opponent_concession = current_opponent_util - self.last_opponent_utility
            if opponent_concession > 0:
                # Math is now done on the local float 'target'
                target -= (opponent_concession / self.strictness)

        self._update_opponent_history(offer, ufun)

        floor_utility = float(ufun.reserved_value)
        max_utility = float(ufun(ufun.extreme_outcomes()[1]))
        
        # Clamp the float and save it back to self
        self.current_target_utility = max(floor_utility, min(max_utility, target))

        return self._find_outcome_for_utility(self.current_target_utility, ufun, nmi)


class StallBreakingBidding(AdaptiveBiddingStrategy):
    def __init__(self, stall_tolerance: int = 5, unblock_bump: float = 0.05):
        super().__init__()
        self.stall_tolerance = stall_tolerance
        self.stall_counter = 0
        self.unblock_bump = unblock_bump

    def generate(self, state: SAOState, ufun: UtilityFunction, nmi: Any) -> Optional[Outcome]:
        # Pylance now knows target is strictly a float
        target: float = self._initialize_target_utility(ufun)
        offer = state.current_offer

        if offer is None or state.step == 0:
            self.current_target_utility = target
            return ufun.extreme_outcomes()[1]

        current_opponent_util = float(ufun(offer))

        if self.last_opponent_offer is not None:
            if current_opponent_util <= self.last_opponent_utility:
                self.stall_counter += 1
            else:
                self.stall_counter = 0

        self._update_opponent_history(offer, ufun)

        if self.stall_counter >= self.stall_tolerance:
            max_utility = float(ufun(ufun.extreme_outcomes()[1]))
            floor_utility = float(ufun.reserved_value)
            range_util = max_utility - floor_utility
            
            # Math is now done on the local float 'target'
            target -= (range_util * self.unblock_bump)
            self.stall_counter = 0

        floor_utility = float(ufun.reserved_value)
        max_utility = float(ufun(ufun.extreme_outcomes()[1]))
        
        # Clamp the float and save it back to self
        self.current_target_utility = max(floor_utility, min(max_utility, target))

        return self._find_outcome_for_utility(self.current_target_utility, ufun, nmi)