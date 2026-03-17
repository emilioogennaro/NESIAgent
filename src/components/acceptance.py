# type: ignore
from abc import ABC, abstractmethod
from typing import Any, Optional
from negmas import Outcome
from negmas.sao import SAOState
from negmas.preferences import UtilityFunction
from .opponent_model import OpponentModel

class AcceptanceStrategy(ABC):
    """Abstract base class for all acceptance strategies."""
    
    def __init__(self, opponent_model: Optional[OpponentModel] = None):
        self.opponent_model = opponent_model
    def evaluate(self, offer: Any, state: Any, ufun: Any) -> bool:
        """Evaluate the offer and return True to accept, False to reject."""
        pass


class StaticThresholdAcceptance(AcceptanceStrategy):
    """A simple strategy that accepts any offer above a fixed utility threshold.
    
    This is a baseline strategy that doesn't adapt to time or context.
    Useful for comparison but often performs poorly in realistic negotiations.
    """
    
    def __init__(self, threshold: float = 0.8, opponent_model: Optional[OpponentModel] = None):
        super().__init__(opponent_model)
        self.threshold = threshold

    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        if offer is None:
            return False
            
        return ufun(offer) >= self.threshold


class AspirationalAcceptance(AcceptanceStrategy):
    """Acceptance strategy based on an aspiration level that decreases over time.
    
    The agent starts with high aspirations and gradually lowers them as the
    negotiation progresses. This models the natural behavior of becoming more
    willing to accept worse deals as deadlines approach.
    
    The aspiration follows a Boulware curve: starts at ideal_utility and 
    decreases quadratically towards the reservation_utility.
    """
    
    def __init__(self, ideal_utility: float = 1.0, reservation_utility: float = 0.3, 
                 e_parameter: float = 2.0):
        """
        Args:
            ideal_utility: The utility we aspire to achieve (typically 1.0)
            reservation_utility: The minimum acceptable utility (walk-away point)
            e_parameter: Controls concession curve shape (higher = more stubborn initially)
        """
        self.ideal_utility = ideal_utility
        self.reservation_utility = reservation_utility
        self.e_parameter = e_parameter

    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        if offer is None:
            return False
        
        # Get relative time (0 = start, 1 = end of negotiation)
        time_progress = state.relative_time if state.relative_time is not None else 0
        
        # Compute Boulware curve aspiration level
        # Formula: reservation + (ideal - reservation) * (1 - time_progress)^e_parameter
        aspiration = (self.reservation_utility + 
                     (self.ideal_utility - self.reservation_utility) * 
                     ((1 - time_progress) ** self.e_parameter))
        
        return ufun(offer) >= aspiration


class OpponentAwareAcceptance(AcceptanceStrategy):
    """Acceptance strategy that adapts based on opponent modeling information.
    
    Uses opponent model to predict opponent behavior and adjust acceptance threshold.
    Becomes more lenient against hardliners and more demanding against conceders.
    """
    
    def __init__(self, base_threshold: float = 0.8, opponent_model: Optional[OpponentModel] = None):
        super().__init__(opponent_model)
        self.base_threshold = base_threshold
    
    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        if offer is None:
            return False
        
        threshold = self.base_threshold
        
        # Adjust threshold based on opponent model
        if self.opponent_model:
            # If opponent is a hardliner, be more willing to accept reasonable offers
            if hasattr(self.opponent_model, 'get_opponent_type'):
                opponent_type = self.opponent_model.get_opponent_type()
                if opponent_type == "hardliner":
                    threshold -= 0.1  # More lenient
                elif opponent_type == "conceder":
                    threshold += 0.05  # More demanding
            
            # If opponent concedes slowly, we might need to be more patient
            if hasattr(self.opponent_model, 'get_concession_rate'):
                concession_rate = self.opponent_model.get_concession_rate()
                if concession_rate < 0.3:  # Hardliner
                    threshold -= 0.05
            
            # If opponent frequently offers this outcome, it might be their target
            if hasattr(self.opponent_model, 'get_offer_frequency'):
                frequency = self.opponent_model.get_offer_frequency(offer)
                if frequency > 0.3:  # Frequently offered
                    threshold -= 0.05  # More likely to be acceptable

            # Estimate opponent utility for this offer (Bayesian model)
            if hasattr(self.opponent_model, 'estimate_utility'):
                est = self.opponent_model.estimate_utility(offer)
                if est > 0.8:
                    threshold += 0.05  # Opponent likely values this offer
                elif est < 0.3:
                    threshold -= 0.05  # Opponent likely dislikes this offer

            # Predict opponent utility trend (strategy model)
            if hasattr(self.opponent_model, 'predict_next_utility'):
                next_time = min(1.0, (state.relative_time or 0.0) + 0.05)
                predicted = self.opponent_model.predict_next_utility(ufun, next_time)
                if predicted > 0.8:
                    threshold += 0.05
                elif predicted < 0.3:
                    threshold -= 0.05

        # Ensure threshold stays within reasonable bounds
        threshold = max(0.1, min(0.95, threshold))
        
        return ufun(offer) >= threshold


class AspirationalAcceptance_Weighted(AcceptanceStrategy):
    """ACnext(α, β) acceptance strategy."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, offer, state, ufun, next_offer=None, our_offers=None):
        if offer is None or next_offer is None:
            return False

        return self.alpha * ufun(offer) + self.beta >= ufun(next_offer)


class AspirationalAcceptance_Proposed(AcceptanceStrategy):
    """Accept if received offer is better than all offers we have proposed so far and next offer."""

    def evaluate(self, offer, state, ufun, next_offer=None, our_offers=None):
        if offer is None:
            return False

        if our_offers is None:
            our_offers = []

        current_utility = ufun(offer)

        # Utilities of our past offers
        our_utils = [ufun(o) for o in our_offers if o is not None]

        # Include next planned offer
        if next_offer is not None:
            our_utils.append(ufun(next_offer))

        if not our_utils:
            return False

        return current_utility > min(our_utils)


class ProgressBasedAcceptance(AcceptanceStrategy):
    """Acceptance strategy that requires showing progress in negotiations.
    
    The agent maintains a history of the best offer received so far.
    An offer is only accepted if it either:
    1. Exceeds a minimum threshold, OR
    2. Represents a significant improvement (progress) over the best previous offer
    
    This prevents the agent from accepting stagnant negotiations and encourages
    the opponent to improve their offers.
    """
    
    def __init__(self, min_threshold: float = 0.5, progress_ratio: float = 1.02):
        """
        Args:
            min_threshold: Minimum utility to accept (hard floor)
            progress_ratio: Each acceptable offer must be at least this factor better 
                          than the previous best (e.g., 1.02 = 2% improvement required)
        """
        self.min_threshold = min_threshold
        self.progress_ratio = progress_ratio
        self.best_offer_utility = None

    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        if offer is None:
            return False
        
        offer_utility = ufun(offer)
        
        # Accept if above minimum threshold
        if offer_utility >= self.min_threshold:
            # Update best offer if this one is better
            if self.best_offer_utility is None or offer_utility > self.best_offer_utility:
                self.best_offer_utility = offer_utility
                # Accept the first reasonably good offer
                return True
            
            # For subsequent offers, require progress
            if offer_utility >= self.best_offer_utility * self.progress_ratio:
                self.best_offer_utility = offer_utility
                return True
        
        return False


class TimeBasedConcessionAcceptance(AcceptanceStrategy):
    """Acceptance strategy that becomes more lenient as time runs out.
    
    The strategy uses a linear concession function: as time progresses from
    start to finish, the acceptance threshold decreases from an initial value
    to a reservation value. This models realistic negotiation behavior where
    agents become increasingly desperate as deadlines approach.
    """
    
    def __init__(self, initial_threshold: float = 0.9, final_threshold: float = 0.4):
        """
        Args:
            initial_threshold: Minimum utility to accept at the start
            final_threshold: Minimum utility to accept near the deadline
        """
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold

    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        if offer is None:
            return False
        
        # Get relative time (0 = start, 1 = end)
        time_progress = state.relative_time if state.relative_time is not None else 0
        
        # Linearly interpolate between initial and final thresholds
        current_threshold = (self.initial_threshold - 
                            (self.initial_threshold - self.final_threshold) * time_progress)
        
        return ufun(offer) >= current_threshold


class AdaptiveAcceptance(AcceptanceStrategy):
    """Acceptance strategy that adapts to opponent's bidding behavior.
    
    This strategy tracks the offers received from the opponent and adjusts
    the acceptance threshold based on:
    1. The average quality of opponent's offers (are they improving?)
    2. The variance in offers (are they being consistent?)
    3. Overall negotiation time progress
    
    If the opponent is consistently offering good deals, accept them sooner.
    If the opponent is being stubborn, adjust expectations downward as time passes.
    """
    
    def __init__(self, base_threshold: float = 0.6, learning_rate: float = 0.1):
        """
        Args:
            base_threshold: Initial acceptance threshold
            learning_rate: How quickly to adapt to opponent's behavior (0-1)
        """
        self.base_threshold = base_threshold
        self.learning_rate = learning_rate
        self.opponent_offer_history = []
        self.adapted_threshold = base_threshold

    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        if offer is None:
            return False
        
        offer_utility = ufun(offer)
        
        # Track opponent's offers
        self.opponent_offer_history.append(offer_utility)
        
        # Adapt threshold based on opponent's average performance
        if len(self.opponent_offer_history) > 1:
            avg_opponent_utility = sum(self.opponent_offer_history) / len(self.opponent_offer_history)
            # Adapt: if opponent is doing well, be less demanding; if not, lower expectations as time passes
            time_progress = state.relative_time if state.relative_time is not None else 0
            self.adapted_threshold = (self.base_threshold * (1 - self.learning_rate * avg_opponent_utility) - 
                                     0.1 * time_progress)
        
        return offer_utility >= max(0.1, self.adapted_threshold)  # Never go below 0.1 utility


class HybridAcceptance(AcceptanceStrategy):
    """Hybrid acceptance strategy combining multiple factors.
    
    This strategy uses a weighted combination of:
    1. Aspiration level (decreases over time)
    2. Opponent model (adapts to what they're offering)
    3. Time pressure (urgent acceptance as deadline approaches)
    
    Offers are accepted if the weighted score exceeds 0.5, creating a
    balanced approach that considers multiple negotiation aspects.
    """
    
    def __init__(self, aspiration_weight: float = 0.4, opponent_weight: float = 0.3, 
                 time_weight: float = 0.3):
        """
        Args:
            aspiration_weight: Weight of aspiration level
            opponent_weight: Weight of opponent modeling
            time_weight: Weight of time-based pressure
        """
        self.aspiration_weight = aspiration_weight
        self.opponent_weight = opponent_weight
        self.time_weight = time_weight
        self.opponent_utilities = []

    def evaluate(self, offer: Outcome, state: SAOState, ufun: UtilityFunction) -> bool:
        if offer is None:
            return False
        
        offer_utility = ufun(offer)
        time_progress = state.relative_time if state.relative_time is not None else 0
        
        # Component 1: Aspiration (Boulware curve)
        aspiration_level = 0.3 + 0.7 * ((1 - time_progress) ** 1.5)
        aspiration_score = min(1.0, offer_utility / aspiration_level) if aspiration_level > 0 else 0
        
        # Component 2: Opponent model (are they giving good offers?)
        self.opponent_utilities.append(offer_utility)
        if len(self.opponent_utilities) > 1:
            avg_opponent_utility = sum(self.opponent_utilities) / len(self.opponent_utilities)
            opponent_score = min(1.0, avg_opponent_utility)
        else:
            opponent_score = 0.5  # Neutral on first offer
        
        # Component 3: Time pressure (more lenient as time runs out)
        time_score = 0.3 + 0.7 * time_progress
        
        # Weighted combination
        decision_score = (self.aspiration_weight * aspiration_score + 
                         self.opponent_weight * opponent_score + 
                         self.time_weight * time_score)
        
        return decision_score >= 0.5