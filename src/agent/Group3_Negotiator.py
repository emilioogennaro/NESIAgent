import random
import numpy as np
import warnings

from negmas.sao import SAONegotiator, ResponseType
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_SAMPLE_SIZE = 300      # random bids to evaluate per propose() call
_GP_REFIT_INTERVAL = 3  # refit GP every N new opponent offers


def _fast_ufun_max(ufun, issue_values):
    """
    Compute ufun.max() in O(sum of issue sizes) instead of O(product of issue sizes).

    LinearAdditiveUtilityFunction decomposes as:
        ufun(o) = sum_i  w_i * f_i(o_i)
    Because issues are independent, the global max is found by maximising
    each issue's contribution separately.  We do this by holding all other
    issues fixed at a baseline and sweeping only the target issue's values.
    """
    n_issues = len(issue_values)
    # Any valid baseline outcome works; midpoint is a safe choice.
    baseline = tuple(vals[len(vals) // 2] for vals in issue_values)

    # Contribution of each issue at the baseline
    baseline_u = ufun(baseline)

    # For each issue, find its best value and accumulate the improvement
    # over the baseline.  Sum baseline_u + all per-issue deltas = global max.
    total_delta = 0.0
    for i, vals in enumerate(issue_values):
        best_u_for_issue = max(
            ufun(baseline[:i] + (v,) + baseline[i + 1:])
            for v in vals
        )
        total_delta += best_u_for_issue - baseline_u

    return baseline_u + total_delta


class Group3_Negotiator(SAONegotiator):
    def __init__(self, name="Group3_Negotiator", alpha=1.0, gamma=0.01,
                 m=0.5, E=0.1, eps=0.02, **kwargs):
        super().__init__(name=name, **kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.m = m
        self.E = E
        self.eps = eps

        self.beta = None
        self._ufun_max = None
        self._issue_values = None

        self.proposed_offers = set()
        self.opponent_offers_history = []
        self.opponent_ufun_predictions_history = []

        self.best_received_offer = None
        self.best_received_utility = float("-inf")

        self._cached_prediction = None
        self._cached_sigma = None
        self._gp_last_fit_at = 0

        self._opponent_class_cache = None
        self._opponent_class_at = 0

        self._ufun_cache = {}

    # ------------------------------------------------------------------
    # Cached utility evaluation
    # ------------------------------------------------------------------

    def _ufun(self, offer):
        """Memoised ufun — avoids re-evaluating the same outcome twice."""
        try:
            return self._ufun_cache[offer]
        except KeyError:
            v = self.ufun(offer)
            self._ufun_cache[offer] = v
            return v

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_negotiation_start(self, state):
        self.beta = self.ufun.reserved_value

        # Build per-issue value lists ONCE up front.
        self._issue_values = [list(issue.all) for issue in self.nmi.issues]

        self._ufun_max = _fast_ufun_max(self.ufun, self._issue_values)
        self.alpha = max(self.alpha, self._ufun_max)

        self.proposed_offers.clear()
        self.opponent_offers_history.clear()
        self.opponent_ufun_predictions_history.clear()
        self.best_received_offer = None
        self.best_received_utility = float("-inf")
        self._cached_prediction = None
        self._cached_sigma = None
        self._gp_last_fit_at = 0
        self._opponent_class_cache = None
        self._opponent_class_at = 0
        self._ufun_cache.clear()

    # ------------------------------------------------------------------
    # Concession curve
    # ------------------------------------------------------------------

    def aspiration(self, state):
        t = state.relative_time
        return max(
            (self.alpha - self.beta) * (1 - self.gamma ** (1 - t) / (1 - self.gamma)) + self.beta,
            self.beta,
        )

    def target_utility(self, state):
        t = state.relative_time
        E = max(self.E, 1e-6)
        g = self.m + (1.0 - self.m) * (1.0 - (t ** (1.0 / E)))
        return max(g, self.ufun.reserved_value)

    # ------------------------------------------------------------------
    # Opponent classification — cached per history length
    # ------------------------------------------------------------------

    def calculate_ubi(self, bids):
        def rec(xs, ubi=0):
            if len(xs) < 2:
                return ubi
            mid = len(xs) // 2
            left, right = xs[:mid], xs[mid:]
            len_left = len(set(map(tuple, left)))
            len_right = len(set(map(tuple, right)))
            if len_left > 0 and len_right > 0 and len_left < len_right:
                return rec(right, ubi + 1)
            return ubi
        return rec(bids, 0)

    def calculate_aui(self, bids):
        utils = [self._ufun(b if isinstance(b, tuple) else tuple(b)) for b in bids]

        def rec(xs, aui=0):
            if len(xs) < 2:
                return aui
            mid = len(xs) // 2
            left, right = xs[:mid], xs[mid:]
            if len(left) > 0 and len(right) > 0 and np.mean(left) < np.mean(right):
                return rec(right, aui + 1)
            return aui
        return rec(utils, 0)

    def classify_opponent(self):
        n = len(self.opponent_offers_history)
        if self._opponent_class_cache is not None and n == self._opponent_class_at:
            return self._opponent_class_cache

        if n < 4:
            result = ("unknown", 0, 0)
        else:
            bids = [o for _, o in self.opponent_offers_history]
            ubi = self.calculate_ubi(bids)
            aui = self.calculate_aui(bids)
            if ubi >= 5:
                result = ("boulware", ubi, aui)
            elif aui <= 2:
                result = ("hardliner", ubi, aui)
            else:
                result = ("conceder", ubi, aui)

        self._opponent_class_cache = result
        self._opponent_class_at = n
        return result

    # ------------------------------------------------------------------
    # Random bid generation — never touches outcome_space iterator
    # ------------------------------------------------------------------

    def _random_offer(self):
        return tuple(random.choice(vals) for vals in self._issue_values)

    def _sample_offers(self, n=_SAMPLE_SIZE):
        seen = self.proposed_offers
        offers = []
        attempts = 0
        max_attempts = n * 4
        while len(offers) < n and attempts < max_attempts:
            o = self._random_offer()
            if o not in seen:
                offers.append(o)
            attempts += 1
        return offers or ([self.best_received_offer] if self.best_received_offer else [])

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    def propose(self, state, **kwargs):
        aspiration = self.aspiration(state)
        goal = self.target_utility(state)
        t = state.relative_time
        delta = (3 * t + 1) * self.eps

        lower = max(self.ufun.reserved_value, goal - delta)
        upper = min(self._ufun_max, goal + delta)

        offers = self._sample_offers()
        if not offers:
            return None

        opp_type, ubi, _ = self.classify_opponent()
        if opp_type == "boulware" and ubi > 0:
            late_round = t > 1.0 - (0.5 / ubi)
            if late_round:
                self.m = min(self.m, 0.3)
                if (self.best_received_offer is not None
                        and self.best_received_utility >= self.m):
                    self.proposed_offers.add(self.best_received_offer)
                    return self.best_received_offer

        scored = [(o, self._ufun(o)) for o in offers]
        acceptable = [(o, u) for o, u in scored if u >= aspiration]

        if acceptable:
            in_band = [(o, u) for o, u in acceptable if lower <= u <= upper]
            if in_band:
                offer, _ = min(in_band, key=lambda x: x[1] - goal)
            else:
                offer, _ = min(acceptable, key=lambda x: x[1] - aspiration)
        else:
            offer, _ = max(scored, key=lambda x: x[1])

        if (self.best_received_offer is not None
                and self.best_received_utility > self._ufun(offer)):
            offer = self.best_received_offer

        self.proposed_offers.add(offer)
        return offer

    # ------------------------------------------------------------------
    # Respond
    # ------------------------------------------------------------------

    def respond(self, state, source=None):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        self.opponent_offers_history.append((state.relative_time, offer))

        offer_u = self._ufun(offer)
        if offer_u > self.best_received_utility:
            self.best_received_offer = offer
            self.best_received_utility = offer_u

        next_offer, sigma = self.opponent_strategy(state)

        snapped = self._snap_to_outcome(next_offer)
        predicted_utility = self._ufun(snapped)
        self.opponent_ufun_predictions_history.append(
            (state.relative_time, predicted_utility)
        )

        sigma_scalar = float(np.mean(sigma))
        threshold = max(self.aspiration(state), predicted_utility - sigma_scalar)

        if offer_u >= threshold:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _snap_to_outcome(self, float_offer):
        """Round GP continuous predictions to nearest valid issue value."""
        return tuple(
            min(vals, key=lambda x: abs(x - float_offer[i]))
            for i, vals in enumerate(self._issue_values)
        )

    # ------------------------------------------------------------------
    # GP opponent modelling — throttled + fixed kernel (no hyper search)
    # ------------------------------------------------------------------

    def opponent_strategy(self, state):
        n = len(self.opponent_offers_history)

        if n < 2:
            last_offer = (
                np.array(self.opponent_offers_history[-1][1])
                if self.opponent_offers_history
                else np.zeros(len(self.nmi.issues))
            )
            return last_offer, np.zeros_like(last_offer)

        if (self._cached_prediction is not None
                and n - self._gp_last_fit_at < _GP_REFIT_INTERVAL):
            return self._cached_prediction, self._cached_sigma

        times = np.array([t for t, o in self.opponent_offers_history]).reshape(-1, 1)
        offers = np.array([o for t, o in self.opponent_offers_history])
        n_issues = offers.shape[1]

        predicted_offer = np.zeros(n_issues)
        sigma = np.zeros(n_issues)

        kernel = (
            RBF(length_scale=0.05, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=1e-2, noise_level_bounds="fixed")
        )

        for i in range(n_issues):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                optimizer=None,
            )
            gp.fit(times, offers[:, i])
            pred, std = gp.predict(
                np.array([[state.relative_time]]), return_std=True
            )
            predicted_offer[i] = pred[0]
            sigma[i] = std[0]

        self._cached_prediction = predicted_offer
        self._cached_sigma = sigma
        self._gp_last_fit_at = n

        return predicted_offer, sigma