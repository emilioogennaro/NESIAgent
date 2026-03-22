from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import collections
import numpy as np
from negmas import SAOState
from negmas.sao import ResponseType, SAONegotiator
from scipy.interpolate import UnivariateSpline

# ============================================================================
# utility/utils.py
# ============================================================================


class UtilityHelper:
    """Shared helper for computing utilities and reading time.

    This is the single place where `preferences` (our utility function) is stored.
    All other BOA components access utilities through this helper rather than
    holding a direct reference to the ufun.
    """

    def __init__(self, preferences):
        self.preferences = preferences

    def get_time(self, state: SAOState) -> float:
        """Returns normalized time (0.0 to 1.0)."""
        return state.relative_time

    def get_utility(self, offer: Optional[Tuple]) -> float:
        """Safely returns the utility of an offer to our agent."""
        if not offer:
            return 0.0
        return float(self.preferences(offer))


# ============================================================================
# acceptance/acceptance_strategy.py
# ============================================================================

FALLBACK_RESERVED_VALUE = 0.0


class AcceptanceStrategy(ABC):
    """Abstract base class for acceptance strategies.

    Subclasses must implement ``should_accept``. The optional
    ``record_proposal`` hook lets strategies that need proposal history
    (e.g. AClow) track bids as they are made.
    """

    def __init__(self, utility_helper: UtilityHelper, reservation_value: float = FALLBACK_RESERVED_VALUE):
        self.utility_helper = utility_helper
        self.reservation_value = reservation_value

    @abstractmethod
    def should_accept(
        self,
        offer: Optional[Tuple],
        state: SAOState,
        my_target_utility: float,
        opp_model,
    ) -> bool:
        """Return True if the opponent's *offer* should be accepted."""
        raise NotImplementedError

    def record_proposal(self, bid: Optional[Tuple]) -> None:
        """Called after each proposal so strategies can track bid history.

        No-op by default; override in strategies that need it.
        """
        return None


# ============================================================================
# acceptance/ac_next.py
# ============================================================================


class ACnext(AcceptanceStrategy):
    """Accept if the opponent's offer is at least as good as the next bid
    we would make ourselves.

    This avoids rejecting an offer and then proposing something no better
    on our next turn. ``my_target_utility`` represents the utility level
    our bidding strategy is currently targeting.
    """

    def should_accept(
        self,
        offer: Optional[Tuple],
        state: SAOState,
        my_target_utility: float,
        opp_model,
    ) -> bool:
        if offer is None:
            return False

        offer_utility = self.utility_helper.get_utility(offer)
        return offer_utility >= my_target_utility


# ============================================================================
# acceptance/ac_low.py
# ============================================================================


class AClow(AcceptanceStrategy):
    """Accept if the opponent's offer is at least as good as the minimum of
    our next planned offer or any offer we have already proposed.

    A safer variant of ACnext, especially when bidding is not monotonic,
    because it avoids rejecting an offer and then later offering something
    equally bad or worse.
    """

    def __init__(self, utility_helper: UtilityHelper, reservation_value: float = 0.0):
        super().__init__(utility_helper, reservation_value)
        self._proposed_utilities: List[float] = []

    def record_proposal(self, bid: Optional[Tuple]) -> None:
        if bid is not None:
            self._proposed_utilities.append(self.utility_helper.get_utility(bid))

    def should_accept(
        self,
        offer: Optional[Tuple],
        state: SAOState,
        my_target_utility: float,
        opp_model,
    ) -> bool:
        if offer is None:
            return False

        offer_utility = self.utility_helper.get_utility(offer)

        if self._proposed_utilities:
            threshold = min(my_target_utility, min(self._proposed_utilities))
        else:
            threshold = my_target_utility

        return offer_utility >= threshold


# ============================================================================
# acceptance/ac_asp.py
# ============================================================================

ASPIRATION_INIT = 0.95
ASPIRATION_EXPONENT = 2.0


class ACasp(AcceptanceStrategy):
    """Aspiration-level acceptance strategy.

    Accept when the opponent's offer utility meets or exceeds a
    time-dependent aspiration level lambda(t). The aspiration starts high
    (strict) and decreases toward the reservation value as the deadline
    approaches, so the agent becomes progressively more willing to accept.
    """

    def __init__(
        self,
        utility_helper: UtilityHelper,
        reservation_value: float = 0.0,
        aspiration_init: float = ASPIRATION_INIT,
        aspiration_exponent: float = ASPIRATION_EXPONENT,
    ):
        super().__init__(utility_helper, reservation_value)
        self.aspiration_init = aspiration_init
        self.aspiration_exponent = aspiration_exponent

    def _aspiration_level(self, t: float) -> float:
        rv = self.reservation_value
        return rv + (1.0 - t ** self.aspiration_exponent) * (self.aspiration_init - rv)

    def should_accept(
        self,
        offer: Optional[Tuple],
        state: SAOState,
        my_target_utility: float,
        opp_model,
    ) -> bool:
        if offer is None:
            return False

        offer_utility = self.utility_helper.get_utility(offer)
        t = self.utility_helper.get_time(state)

        return offer_utility >= self._aspiration_level(t)


# ============================================================================
# acceptance/ac_exp.py
# ============================================================================

# Base aspiration settings
INITIAL_THRESHOLD = 0.90
AC_BETA = 1.5

# Endgame handling
ENDGAME_T = 0.95
ENDGAME_SLACK = 0.04   # accept if within this of the best offer seen so far

# Opponent-model adjustment
MAX_OPP_BONUS = 0.03   # smaller because the opponent model is still noisy
STAGNATION_WINDOW = 4
STAGNATION_EPS = 0.02
STAGNATION_BONUS = 0.02

# ACnext-ish margin
NEXT_MARGIN = 1e-6


class ACexp(AcceptanceStrategy):
    """
    Hybrid acceptance strategy:
    - Reservation-value safe
    - ACasp base (time-dependent aspiration threshold)
    - ACnext-ish guard using my_target_utility
    - Opponent-model concession bonus
    - Best-seen endgame safeguard
    """

    def __init__(self, utility_helper: UtilityHelper, reservation_value: float = 0.0):
        super().__init__(utility_helper, reservation_value)

        self.best_offer_utility_seen = -1.0
        self.best_offer_seen = None

        # Track predicted opponent utility of their own offers over time
        self.opp_pred_history: Deque[float] = deque(maxlen=STAGNATION_WINDOW)
        self.best_opp_pred_seen = 0.0

    def _aspiration_threshold(self, t: float, rv: float) -> float:
        """ACasp-style power-curve threshold."""
        return INITIAL_THRESHOLD - (INITIAL_THRESHOLD - rv) * (t ** AC_BETA)

    def _opponent_bonus(self, current_opp_pred: float) -> float:
        """
        More bonus when the opponent seems to have conceded more
        relative to the best-for-them offers they made before.
        """
        concession = max(0.0, self.best_opp_pred_seen - current_opp_pred)
        return min(MAX_OPP_BONUS, concession)

    def _stagnating(self) -> bool:
        """Did the opponent stop making meaningfully different offers?"""
        if len(self.opp_pred_history) < self.opp_pred_history.maxlen:
            return False
        return (max(self.opp_pred_history) - min(self.opp_pred_history)) < STAGNATION_EPS

    def should_accept(
        self,
        offer: Optional[Tuple],
        state: SAOState,
        my_target_utility: float,
        opp_model,
    ) -> bool:
        if offer is None:
            return False

        own_u = self.utility_helper.get_utility(offer)
        rv = self.reservation_value
        t = self.utility_helper.get_time(state)

        # Never accept below walk-away value
        if own_u < rv:
            return False

        # Track best offer we have ever seen from the opponent
        if own_u > self.best_offer_utility_seen:
            self.best_offer_utility_seen = own_u
            self.best_offer_seen = offer

        # Base threshold = ACasp
        threshold = max(rv, self._aspiration_threshold(t, rv))

        # ACnext-ish improvement:
        # if the current offer is at least as good as what we currently aim to get,
        # don't reject it just to send something worse/equivalent.
        if my_target_utility is not None:
            threshold = min(threshold, max(rv, float(my_target_utility) - NEXT_MARGIN))

        # Opponent-model signal:
        # if this offer seems worse for them than their earlier favorite offers,
        # they are likely conceding -> we can lower our threshold a bit.
        opp_pred = 0.5
        if opp_model is not None:
            try:
                opp_pred = float(opp_model.predict_utility(offer))
            except Exception:
                opp_pred = 0.5

        self.opp_pred_history.append(opp_pred)
        self.best_opp_pred_seen = max(self.best_opp_pred_seen, opp_pred)

        threshold = max(rv, threshold - self._opponent_bonus(opp_pred))

        # If opponent seems to have stopped changing much and we're already late,
        # accept a bit earlier to avoid pointless deadlock.
        if t >= 0.7 and self._stagnating():
            threshold = max(rv, threshold - STAGNATION_BONUS)

        # Endgame:
        # do not reject something that is basically as good as the best offer seen.
        if t >= ENDGAME_T:
            return own_u >= max(rv, self.best_offer_utility_seen - ENDGAME_SLACK)

        return own_u >= threshold


ACCEPTANCE_STRATEGIES = {
    "ACnext": ACnext,
    "AClow": AClow,
    "ACasp": ACasp,
    "ACexp": ACexp,
}

# ============================================================================
# bidding/wavelet_decomposition.py
# ============================================================================


class WaveletDecomposition:
    """Predict opponent concession trend using DWT denoising + spline extrapolation.

    Based on OMAC (Chen & Weiss, 2012). Uses Daubechies-10 wavelet to extract the
    smooth trend from the opponent's utility time-series, then extends it with a
    cubic smoothing spline for future prediction. Falls back to EWMA when fewer than
    ``min_samples_for_dwt`` observations are available.

    Raw observations are binned into equal time intervals (max utility per interval)
    to reduce noise and computation, following the paper's preprocessing step.
    """

    def __init__(
        self,
        wavelet: str = "db10",
        min_samples_for_dwt: int = 20,
        decomposition_level=None,
        n_intervals: int = 100,
        max_forecast_horizon: float = 0.15,
    ):
        self.wavelet = wavelet
        self.min_samples_for_dwt = min_samples_for_dwt
        self.decomposition_level = decomposition_level

        # Interval binning: divide [0, 1] into n_intervals, track max utility per bin
        self.n_intervals = n_intervals
        self._bins: List[Optional[float]] = [None] * n_intervals  # max utility per interval

        # Max forecast horizon (fraction of [0,1] time) to prevent wild extrapolation
        self.max_forecast_horizon = max_forecast_horizon

        # Raw observations (kept for mean/access by bidding strategy)
        self.times: List[float] = []
        self.utilities: List[float] = []

        self._ewma_alpha = 0.3

    def add_observation(self, t: float, utility: float) -> None:
        """Record one (time, utility-we-received) sample."""
        # Store raw observation
        if self.times and t <= self.times[-1]:
            self.times[-1] = t
            self.utilities[-1] = utility
        else:
            self.times.append(t)
            self.utilities.append(utility)

        # Update interval bin with max utility (paper Section 4.1)
        bin_idx = min(int(t * self.n_intervals), self.n_intervals - 1)
        if self._bins[bin_idx] is None or utility > self._bins[bin_idx]:
            self._bins[bin_idx] = utility

    def get_expected_received_utility(self, t_future: float) -> float:
        """Extrapolate the denoised trend to *t_future* (paper Eq. 8)."""
        binned_times, binned_utils = self._get_binned_series()

        if len(binned_utils) < 2:
            return self.utilities[-1] if self.utilities else 0.5

        # Limit forecast horizon to prevent wild extrapolation (paper: ζ intervals)
        max_t = binned_times[-1] + self.max_forecast_horizon
        t_clamped = min(t_future, max_t)

        # Step 1: Denoise the binned series to get the smooth trend υ
        smoothed = self._extract_smooth_trend(np.array(binned_utils))
        times = np.array(binned_times)

        # Step 2: Compute ratio scaling (paper Eq. 8)
        # E_ru(t) = α(t) * (1 + Stdev(ratio_between_χ_and_υ))
        raw = np.array(binned_utils)
        ratio_stdev = self._compute_ratio_stdev(raw, smoothed)

        # Step 3: Fit smoothing spline and extrapolate
        k = min(3, len(times) - 1)
        try:
            s = len(times) * 0.01
            spline = UnivariateSpline(times, smoothed, k=k, s=s)
            predicted = float(spline(t_clamped))
        except Exception:
            dt = times[-1] - times[0]
            slope = (smoothed[-1] - smoothed[0]) / (dt + 1e-9)
            predicted = float(smoothed[-1] + slope * (t_clamped - times[-1]))

        # Apply ratio scaling: account for gap between smooth trend and raw peaks
        predicted *= (1.0 + ratio_stdev)

        return float(np.clip(predicted, 0.0, 1.0))

    def _get_binned_series(self) -> Tuple[List[float], List[float]]:
        """Return (times, max_utilities) for filled interval bins."""
        times: List[float] = []
        utils: List[float] = []
        interval_width = 1.0 / self.n_intervals
        for i, val in enumerate(self._bins):
            if val is not None:
                times.append((i + 0.5) * interval_width)
                utils.append(val)
        return times, utils

    def _compute_ratio_stdev(self, raw: np.ndarray, smoothed: np.ndarray) -> float:
        """Standard deviation of ratio between raw signal χ and smooth trend υ (paper Eq. 8)."""
        safe_smoothed = np.where(np.abs(smoothed) < 1e-9, 1e-9, smoothed)
        ratios = raw / safe_smoothed
        return float(np.std(ratios))

    def _extract_smooth_trend(self, utilities: np.ndarray) -> np.ndarray:
        """DWT denoising (or EWMA fallback) of the utility series."""
        if len(utilities) < self.min_samples_for_dwt:
            return self._ewma_fallback(utilities)

        import pywt

        level = self.decomposition_level
        if level is None:
            max_level = pywt.dwt_max_level(len(utilities), pywt.Wavelet(self.wavelet).dec_len)
            level = max(1, max_level - 1)

        coeffs = pywt.wavedec(utilities, self.wavelet, level=level)
        for i in range(1, len(coeffs)):
            coeffs[i] = np.zeros_like(coeffs[i])

        smoothed = pywt.waverec(coeffs, self.wavelet)
        return smoothed[: len(utilities)]

    def _ewma_fallback(self, utilities: np.ndarray) -> np.ndarray:
        """Exponentially-weighted moving average for early rounds."""
        smoothed = np.empty_like(utilities)
        smoothed[0] = utilities[0]
        a = self._ewma_alpha
        for i in range(1, len(utilities)):
            smoothed[i] = a * utilities[i] + (1 - a) * smoothed[i - 1]
        return smoothed


# ============================================================================
# opponent_model/opponent_model.py
# ============================================================================

# Section 5: Default Parameters (Optimized v2.1)
ETA = 0.05           # Lower forgetting rate for long-term strategic trends
ALPHA_LAPLACE = 0.1  # Laplace smoothing constant for categorical issues
H_MIN = 0.05         # Higher minimum bandwidth to generalize utility
ALPHA = 0.1          # Base learning rate for weight stability
MAX_ROUNDS = 10000   # Pre-allocation limit for O(1) array updates


def numeric_distance_vec(v, history_vec):
    """Vectorized Numeric Distance on pre-normalized inputs."""
    return np.abs(v - history_vec)


def gaussian_kernel_vec(z_vec):
    """Vectorized Gaussian Kernel (unscaled to peak at 1.0 for utility mapping)."""
    return np.exp(-(z_vec ** 2) / 2)


def get_adaptive_bandwidth(history_values, h_min=H_MIN):
    """Adaptive Bandwidth h_{i,t} using Silverman’s Rule of Thumb with clamping."""
    t = len(history_values)
    if t < 2:
        return 0.5

    sigma = max(np.std(history_values), 1e-6)
    bw = ((4 / (3 * t)) ** 0.2) * sigma
    return max(bw, h_min)


def get_exponential_forgetting_weights(times_vec, t, eta=ETA):
    """Vectorized recency weights alpha_k = exp(-eta * (t - k))."""
    return np.exp(-eta * (t - times_vec))


class OpponentModel:
    """
    Recency-Aware KDE Opponent Model v2.1.
    Optimized for computational efficiency via pre-allocated NumPy arrays.
    """

    def __init__(self, nmi):
        self.nmi = nmi
        self.issue_names = list(nmi.outcome_space.issue_names)
        self.num_issues = len(self.issue_names)

        # Pre-allocate NumPy arrays for O(1) updates
        self.h_vals_arr = np.zeros((self.num_issues, MAX_ROUNDS))
        self.h_times_arr = np.zeros((self.num_issues, MAX_ROUNDS))

        self.n_bids_seen = 0
        self.alpha = ALPHA

        # Cached scalars for performance
        self.cached_bandwidths = np.full(self.num_issues, 0.5)

        # Issue Domains using NegMAS API properties
        self.issue_types: Dict[int, str] = {}
        self.issue_domains: Dict[int, Any] = {}
        self.issue_value_maps: Dict[int, Dict[Any, int]] = {}

        for i, issue in enumerate(nmi.outcome_space.issues):
            if issue.is_numeric:
                self.issue_types[i] = "numeric"
                self.issue_domains[i] = (float(issue.min_value), float(issue.max_value))
            else:
                self.issue_types[i] = "categorical"
                issue_values = (
                    list(issue.all_values)
                    if hasattr(issue, "all_values")
                    else (list(issue.values) if hasattr(issue, "values") else [])
                )
                self.issue_value_maps[i] = {val: idx for idx, val in enumerate(issue_values)}
                domain_size = len(issue_values)
                self.issue_domains[i] = max(domain_size, 1)

        # Initial Weights (Uniform)
        self.issue_weights = np.full(self.num_issues, 1.0 / self.num_issues)

        # History tracking for BiddingStrategy
        self.bid_history: List[Tuple] = []

        # Strategy Modeling (v3)
        self.recent_utilities = collections.deque(maxlen=10)
        self.dynamic_eta = ETA
        self.max_anchor_density = 0.0
        self.first_seen = [{} for _ in range(self.num_issues)]
        self.baseline_offer = None

    def update(self, offer: Optional[Tuple], state) -> None:
        """Update internal state and learn issue weights."""
        if not offer:
            return

        self.bid_history.append(offer)

        # 1. Normalize/Map the offer upon ingestion
        norm_offer = []
        for i in range(self.num_issues):
            val = offer[i]
            if self.issue_types[i] == "numeric":
                min_v, max_v = self.issue_domains[i]
                norm_val = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0
                norm_offer.append(norm_val)
            else:
                mapped_val = self.issue_value_maps[i].get(val, 0)
                norm_offer.append(float(mapped_val))
        norm_offer = tuple(norm_offer)

        # 1.1 Lock in the Opening Bid (v3.3)
        if self.n_bids_seen == 0:
            self.baseline_offer = norm_offer

        # 1.2 Discovery Tracking (v3.2)
        for i in range(self.num_issues):
            val = norm_offer[i]
            if self.issue_types[i] == "categorical":
                if val not in self.first_seen[i]:
                    self.first_seen[i][val] = self.n_bids_seen

        # 2. Weight Learning (Time-dependent Alpha)
        if self.n_bids_seen > 0:
            prev_idx = self.n_bids_seen - 1
            effective_alpha = self.alpha / np.sqrt(self.n_bids_seen)

            for i in range(self.num_issues):
                prev_val = self.h_vals_arr[i, prev_idx]
                if self.issue_types[i] == "numeric":
                    dist = abs(norm_offer[i] - prev_val)
                else:
                    dist = 0.0 if norm_offer[i] == prev_val else 1.0

                stability = 1.0 - dist
                self.issue_weights[i] = (effective_alpha * stability) + (
                    (1.0 - effective_alpha) * self.issue_weights[i]
                )

            total_w = np.sum(self.issue_weights)
            if total_w > 0:
                self.issue_weights /= total_w

        # 3. Store in Pre-allocated Arrays (O(1) insertion)
        idx = self.n_bids_seen

        if idx >= self.h_vals_arr.shape[1]:
            pad = np.zeros((self.num_issues, MAX_ROUNDS))
            self.h_vals_arr = np.hstack((self.h_vals_arr, pad))
            self.h_times_arr = np.hstack((self.h_times_arr, pad))

        t_val = self.n_bids_seen + 1  # 1-indexed time

        for i in range(self.num_issues):
            self.h_vals_arr[i, idx] = norm_offer[i]
            self.h_times_arr[i, idx] = t_val

            if self.issue_types[i] == "numeric":
                active_history = self.h_vals_arr[i, : idx + 1]
                self.cached_bandwidths[i] = get_adaptive_bandwidth(active_history)

        self.n_bids_seen += 1

        # 4. Strategy Modeling (Change-Point Detection) (v3)
        current_util = self.predict_utility(offer)

        if len(self.recent_utilities) == 10:
            diffs = np.diff(list(self.recent_utilities))
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)

            current_diff = current_util - self.recent_utilities[-1]

            if abs(current_diff - mean_diff) > 2 * max(std_diff, 0.01):
                self.dynamic_eta = ETA * 5.0
            else:
                self.dynamic_eta = max(ETA, self.dynamic_eta * 0.9)

        self.recent_utilities.append(current_util)

    def predict_utility(self, offer: Optional[Tuple]) -> float:
        """Vectorized estimation of the total utility of an offer."""
        if not offer or self.n_bids_seen == 0:
            return 0.5

        t = self.n_bids_seen

        # 1. Pre-normalize offer and calculate Baseline Similarity (v3.3)
        norm_offer = []
        baseline_similarity = 0.0
        uniform_w = 1.0 / self.num_issues

        for i in range(self.num_issues):
            val = offer[i]
            if self.issue_types[i] == "numeric":
                min_v, max_v = self.issue_domains[i]
                norm_val = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0
            else:
                norm_val = float(self.issue_value_maps[i].get(val, 0))
            norm_offer.append(norm_val)

            if self.baseline_offer is not None:
                if self.issue_types[i] == "numeric":
                    sim = 1.0 - abs(norm_val - self.baseline_offer[i])
                else:
                    sim = 1.0 if norm_val == self.baseline_offer[i] else 0.0
                baseline_similarity += uniform_w * sim

        # 2. Identify Top 2 Issues for Multivariate KDE (v3)
        top_2_idx = []
        if self.num_issues >= 2:
            top_2_idx = np.argsort(self.issue_weights)[-2:]

        total_util = 0.0
        processed_issues = set()

        # Vectorized recency weights (using dynamic_eta)
        h_times_ref = self.h_times_arr[0, :t]
        alpha_vec = get_exponential_forgetting_weights(h_times_ref, t, eta=self.dynamic_eta)
        total_weight = np.sum(alpha_vec)
        dynamic_laplace = ALPHA_LAPLACE * (total_weight / t)

        # 3. Handle 2D Multivariate KDE for Top 2 Numeric Issues
        if len(top_2_idx) == 2:
            idx_A, idx_B = top_2_idx[0], top_2_idx[1]
            if self.issue_types[idx_A] == "numeric" and self.issue_types[idx_B] == "numeric":
                val_A, val_B = norm_offer[idx_A], norm_offer[idx_B]
                h_A, h_B = self.h_vals_arr[idx_A, :t], self.h_vals_arr[idx_B, :t]

                d_sq = (val_A - h_A) ** 2 + (val_B - h_B) ** 2
                h_joint = (self.cached_bandwidths[idx_A] + self.cached_bandwidths[idx_B]) / 2.0

                kernel_2d = np.exp(-d_sq / (2 * h_joint ** 2))

                numerator = np.sum(alpha_vec * kernel_2d) + (dynamic_laplace * 0.5)
                denominator = total_weight + dynamic_laplace
                evaluation = numerator / denominator

                total_util += (self.issue_weights[idx_A] + self.issue_weights[idx_B]) * evaluation
                processed_issues.update([idx_A, idx_B])

        # 4. Fallback to 1D for remaining issues
        for i in range(self.num_issues):
            if i in processed_issues:
                continue

            val = norm_offer[i]
            weight = self.issue_weights[i]
            h_vals = self.h_vals_arr[i, :t]

            if self.issue_types[i] == "numeric":
                z_vec = numeric_distance_vec(val, h_vals) / self.cached_bandwidths[i]
                kernel_vec = gaussian_kernel_vec(z_vec)

                numerator = np.sum(alpha_vec * kernel_vec) + (dynamic_laplace * 0.5)
                denominator = total_weight + dynamic_laplace
                evaluation = numerator / denominator
            else:
                match_vec = (h_vals == val).astype(float)
                domain_size = self.issue_domains[i]

                numerator = np.sum(alpha_vec * match_vec) + dynamic_laplace
                denominator = total_weight + (dynamic_laplace * domain_size)
                evaluation = numerator / denominator

                discovery_round = self.first_seen[i].get(val, t)
                discovery_penalty = np.exp(-0.02 * discovery_round)
                evaluation *= discovery_penalty

            total_util += weight * evaluation

        # 5. Anchor Lock (v3.1) - Fix "Frequency Trap"
        if self.n_bids_seen < 10:
            if total_util > self.max_anchor_density:
                self.max_anchor_density = total_util
        else:
            if self.max_anchor_density > 0:
                total_util = total_util / self.max_anchor_density

        # 6. Uniform Anchor Ceiling (v3.3) - Fix "Weight Shift Trap"
        if self.baseline_offer is not None and t > 5:
            ceiling = min(1.0, baseline_similarity + 0.15)
            total_util = min(total_util, ceiling)

        return float(np.clip(total_util, 0.0, 1.0))

    def get_profile(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the current opponent model."""
        return {
            "issue_weights": dict(zip(self.issue_names, self.issue_weights)),
            "n_bids": self.n_bids_seen,
        }


# ============================================================================
# bidding/bidding_strategy.py
# ============================================================================


class BiddingStrategy:
    """Three-phase Pareto-frontier walker with OMAC backbone and wavelet modulation."""

    # Default phase boundaries (may be adapted dynamically)
    EXPLORATION_END = 0.15
    CONCESSION_START = 0.75

    # OMAC concession parameters (Boulware-style: concede slowly)
    BETA = 0.12
    DELTA = 0.95
    ETA = 0.88

    # Bid rotation: cycle through top-K candidates to avoid repeating the same bid
    ROTATION_POOL_SIZE = 5

    def __init__(self, nmi, utility_helper: UtilityHelper, opponent_model: OpponentModel, reservation_value: float = 0.4):
        self.nmi = nmi
        self.utility_helper = utility_helper
        self.opponent_model = opponent_model
        self.reservation_value = reservation_value

        # Wavelet-based trend predictor for the opponent's concession curve
        self.wavelet = WaveletDecomposition()

        self._outcomes = None
        self._max_utility = None
        self._exploration_bids = None
        self._pareto_bids = None
        self._round_counter = 0
        self._last_bid = None

        # Opponent profile cache (updated every propose() call)
        self._current_opp_profile = None

        # Rejection tracking
        self._rejection_streak = 0

        # Bid rotation: recent bids deque for diversity
        self._recent_bids = deque(maxlen=self.ROTATION_POOL_SIZE)

        # Domain competitiveness cache — computed once
        self._competitiveness = None

    # ------------------------------------------------------------------ #
    # Rejection / acceptance feedback
    # ------------------------------------------------------------------ #

    def record_rejection(self):
        """Called when the opponent implicitly rejects our last bid (counter-offers)."""
        self._rejection_streak += 1

    def record_acceptance(self):
        """Called when the opponent accepts our offer."""
        self._rejection_streak = 0

    # ------------------------------------------------------------------ #
    # Adaptive helpers
    # ------------------------------------------------------------------ #

    def _get_model_confidence(self) -> float:
        """Estimate opponent model confidence from profile issue weights."""
        profile = self._current_opp_profile
        if not profile or not profile.get("issue_weights"):
            return 0.0

        weights = list(profile["issue_weights"].values())
        if not weights:
            return 0.0

        n = len(weights)
        if n <= 1:
            return 0.0
        uniform = 1.0 / n
        mad = sum(abs(w - uniform) for w in weights) / n
        max_mad = (1.0 - uniform)
        return min(1.0, mad / max_mad) if max_mad > 0 else 0.0

    def _get_dynamic_exploration_end(self, confidence: float) -> float:
        """Adaptive exploration boundary."""
        return max(0.10, self.EXPLORATION_END * (1.0 - confidence))

    def _get_dynamic_concession_start(self, confidence: float) -> float:
        """Adaptive concession boundary."""
        return min(0.90, self.CONCESSION_START + 0.15 * confidence)

    def _compute_competitiveness(self):
        """Pearson correlation between our utility and predicted opponent utility."""
        if self._competitiveness is not None and self._round_counter % 20 != 0:
            return self._competitiveness

        if len(self.opponent_model.bid_history) < 10:
            return 0.0

        outcomes = self._get_outcomes()
        if len(outcomes) < 10:
            self._competitiveness = 0.0
            return 0.0

        our = np.array([self.utility_helper.get_utility(o) for o in outcomes])
        opp = np.array([self.opponent_model.predict_utility(o) for o in outcomes])

        if np.std(our) < 1e-9 or np.std(opp) < 1e-9:
            self._competitiveness = 0.0
        else:
            self._competitiveness = float(np.corrcoef(our, opp)[0, 1])

        return self._competitiveness

    # ------------------------------------------------------------------ #
    # Outcome space
    # ------------------------------------------------------------------ #

    def _get_outcomes(self):
        """Lazy-load all (or sampled) outcomes and cache max utility."""
        if self._outcomes is not None:
            return self._outcomes

        self._outcomes = list(self.nmi.outcome_space.enumerate_or_sample(max_cardinality=10_000))
        if self._outcomes:
            self._max_utility = max(self.utility_helper.get_utility(o) for o in self._outcomes)
        else:
            self._max_utility = 1.0

        return self._outcomes

    # ------------------------------------------------------------------ #
    # Bid selection
    # ------------------------------------------------------------------ #

    def _find_bid_near_target(self, target, t: float):
        """Pick a Pareto bid near *target* that maximises predicted opponent utility."""
        pool = self._pareto_bids if self._pareto_bids else self._get_outcomes()

        tolerance = 0.05
        candidates = [b for b in pool if abs(self.utility_helper.get_utility(b) - target) <= tolerance]

        if not candidates:
            tolerance = 0.10
            candidates = [b for b in pool if abs(self.utility_helper.get_utility(b) - target) <= tolerance]

        if not candidates:
            by_dist = sorted(pool, key=lambda b: abs(self.utility_helper.get_utility(b) - target))
            candidates = by_dist[:5]

        recent_set = set(self._recent_bids)

        candidates.sort(key=lambda b: self.opponent_model.predict_utility(b), reverse=True)

        novel = [b for b in candidates if b not in recent_set]

        if novel:
            chosen = novel[0]
        else:
            idx = self._round_counter % len(candidates)
            chosen = candidates[idx]

        self._recent_bids.append(chosen)
        return chosen

    # ------------------------------------------------------------------ #
    # Exploration pool
    # ------------------------------------------------------------------ #

    def _build_exploration_bids(self):
        """Top-20 % utility bids, subsampled for diversity."""
        if self._exploration_bids is not None:
            return self._exploration_bids

        outcomes = self._get_outcomes()
        scored = [(o, self.utility_helper.get_utility(o)) for o in outcomes]
        scored.sort(key=lambda x: x[1], reverse=True)

        top_count = max(1, len(scored) // 5)
        top_bids = scored[:top_count]

        max_exploration = min(30, len(top_bids))
        if len(top_bids) <= max_exploration:
            selected = [b for b, _ in top_bids]
        else:
            step = len(top_bids) / max_exploration
            selected = [top_bids[int(i * step)][0] for i in range(max_exploration)]

        self._exploration_bids = selected
        return self._exploration_bids

    # ------------------------------------------------------------------ #
    # Pareto frontier
    # ------------------------------------------------------------------ #

    def _compute_pareto_frontier(self):
        """Approximate Pareto set using predicted opponent utility."""
        outcomes = self._get_outcomes()

        scored = [
            (
                o,
                self.utility_helper.get_utility(o),
                self.opponent_model.predict_utility(o),
            )
            for o in outcomes
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        pareto = []
        best_opp = -1.0
        for bid, _my_u, opp_u in scored:
            if opp_u > best_opp:
                pareto.append(bid)
                best_opp = opp_u

        if not pareto:
            pareto = [scored[0][0]] if scored else self._outcomes[:1]

        self._pareto_bids = pareto

    # ------------------------------------------------------------------ #
    # Target
    # ------------------------------------------------------------------ #

    def get_current_target(self, state: SAOState) -> float:
        """OMAC-style target utility with several adaptive adjustments."""
        t = self.utility_helper.get_time(state)
        rv = self.reservation_value
        self._get_outcomes()
        max_u = self._max_utility or 1.0

        comp = self._compute_competitiveness()
        ceiling_scale = 1.0 + 0.05 * comp
        ceiling = max_u * (self.DELTA ** self.ETA) * ceiling_scale

        if t < 1e-9:
            target = ceiling
        else:
            target = rv + (1.0 - t ** (1.0 / self.BETA)) * (ceiling - rv)

        exploration_end = self._get_dynamic_exploration_end(self._get_model_confidence())
        if t >= exploration_end and len(self.wavelet.utilities) >= 3:
            expected_opp = self.wavelet.get_expected_received_utility(min(t + 0.05, 1.0))
            baseline = float(np.mean(self.wavelet.utilities))
            adjustment = (expected_opp - baseline) * 0.2
            target += adjustment

        if self._rejection_streak >= 3:
            rejection_nudge = min(0.05, 0.01 * self._rejection_streak)
            target -= rejection_nudge

        if t > 0.95:
            endgame_factor = 1.0 - 4.0 * (t - 0.95)
            target *= endgame_factor

        return float(np.clip(target, rv, max_u))

    # ------------------------------------------------------------------ #
    # Bid proposal
    # ------------------------------------------------------------------ #

    def propose(self, state: SAOState, opp_profile: dict) -> Optional[Tuple]:
        """Phase-routing entry point — called by the negotiator each turn."""
        self._round_counter += 1
        self._current_opp_profile = opp_profile
        t = self.utility_helper.get_time(state)

        confidence = self._get_model_confidence()
        exploration_end = self._get_dynamic_exploration_end(confidence)

        # Phase 1: Exploration
        if t < exploration_end:
            bids = self._build_exploration_bids()
            if not bids:
                return None

            sorted_bids = sorted(bids, key=lambda b: self.opponent_model.predict_utility(b), reverse=True)
            recent_set = set(self._recent_bids)

            novel = [b for b in sorted_bids if b not in recent_set]
            best = novel[0] if novel else sorted_bids[self._round_counter % len(sorted_bids)]
            self._recent_bids.append(best)
            self._last_bid = best
            return best

        if self._pareto_bids is None or self._rejection_streak >= 5:
            self._compute_pareto_frontier()

        target = self.get_current_target(state)
        bid = self._find_bid_near_target(target, t)

        self._last_bid = bid
        return bid


# ============================================================================
# Group4_Negotiator.py
# ============================================================================

DEFAULT_ACCEPTANCE_STRATEGY = "ACexp"


class Group4_Negotiator(SAONegotiator):
    def __init__(
        self,
        *args,
        acceptance_strategy: str = DEFAULT_ACCEPTANCE_STRATEGY,
        reservation_value: float = 0.4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._acceptance_strategy_name = acceptance_strategy
        self._reservation_value = reservation_value
        self.utility_helper = None
        self.opponent_model = None
        self.bidding = None
        self.acceptance = None

    def on_preferences_changed(self, changes):
        """
        Called by NegMAS right before the negotiation starts
        """
        super().on_preferences_changed(changes)

        self.utility_helper = UtilityHelper(self.preferences)

        self.opponent_model = OpponentModel(self.nmi)
        self.bidding = BiddingStrategy(
            self.nmi,
            self.utility_helper,
            self.opponent_model,
            reservation_value=self._reservation_value,
        )

        strategy_cls = ACCEPTANCE_STRATEGIES[self._acceptance_strategy_name]
        self.acceptance = strategy_cls(
            self.utility_helper,
            reservation_value=self.bidding.reservation_value,
        )

    def respond(self, state: SAOState, source: Optional[str] = None) -> ResponseType:
        """
        Called when the opponent makes an offer.
        """
        offer = state.current_offer
        self.opponent_model.update(offer, state)

        # The opponent making a counter-offer means they rejected our last bid
        if self.bidding._last_bid is not None:
            self.bidding.record_rejection()

        # Feed opponent offer into wavelet predictor
        t = self.utility_helper.get_time(state)
        opp_offer_utility = self.utility_helper.get_utility(offer)
        self.bidding.wavelet.add_observation(t, opp_offer_utility)

        my_target_utility = self.bidding.get_current_target(state)

        if self.acceptance.should_accept(offer, state, my_target_utility, self.opponent_model):
            self.bidding.record_acceptance()
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def propose(self, state: SAOState, dest: Optional[str] = None) -> Optional[Tuple]:
        """
        Called when it is our turn to make an offer.
        """
        opp_profile = self.opponent_model.get_profile()
        my_bid = self.bidding.propose(state, opp_profile)

        self.acceptance.record_proposal(my_bid)

        return my_bid

    def on_negotiation_start(self, state: SAOState) -> None:
        """Optional: Setup counters or print statements."""
        pass

    def on_negotiation_end(self, state: SAOState) -> None:
        """Optional: Dump stats to a file for Evaluation report."""
        pass


__all__ = [
    "UtilityHelper",
    "AcceptanceStrategy",
    "ACnext",
    "AClow",
    "ACasp",
    "ACexp",
    "ACCEPTANCE_STRATEGIES",
    "WaveletDecomposition",
    "OpponentModel",
    "BiddingStrategy",
    "Group4_Negotiator",
]