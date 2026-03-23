import random
import numpy as np
import warnings

from negmas.sao import SAONegotiator, ResponseType
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class MyNegotiatorTB_V2(SAONegotiator):
    def __init__(self, name="MyNegotiatorTB", alpha=1.0, gamma=0.01,
                 m=0.5, E=0.1, eps=0.02, **kwargs):
        super().__init__(name=name, **kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.m = m          
        self.E = E         
        self.eps = eps   

        self.proposed_offers = set()
        self.beta = None

        self.opponent_offers_history = []
        self.opponent_ufun_predictions_history = []

        self.best_received_offer = None
        self.best_received_utility = float("-inf")


    def on_negotiation_start(self, state):
        self.beta = self.ufun.reserved_value
        self.alpha = max(self.alpha, self.ufun.max())

        self.proposed_offers.clear()
        self.opponent_offers_history.clear()
        self.opponent_ufun_predictions_history.clear()
        self.best_received_offer = None
        self.best_received_utility = float("-inf")

    def aspiration(self, state):
        t = state.relative_time
        return max(
            (self.alpha - self.beta) * (1 - self.gamma ** (1 - t) / (1 - self.gamma)) + self.beta,
            self.beta
        )

    def target_utility(self, state):
        t = state.relative_time
        E = max(self.E, 1e-6)
        g = self.m + (1.0 - self.m) * (1.0 - (t ** (1.0 / E)))
        return max(g, self.ufun.reserved_value)
  
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
        utils = [
            self.ufun(tuple(b)) if not isinstance(b, tuple) else self.ufun(b)
            for b in bids
        ]

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
        if len(self.opponent_offers_history) < 4:
            return "unknown", 0, 0

        bids = [o for _, o in self.opponent_offers_history]
        ubi = self.calculate_ubi(bids)
        aui = self.calculate_aui(bids)

        if ubi >= 5:
            return "boulware", ubi, aui
        elif aui <= 2:
            return "hardliner", ubi, aui
        else:
            return "conceder", ubi, aui

  
    def propose(self, state, **kwargs):
        aspiration = self.aspiration(state)
        goal = self.target_utility(state)
        t = state.relative_time
        delta = (3 * t + 1) * self.eps

        lower = max(self.ufun.reserved_value, goal - delta)
        upper = min(self.ufun.max(), goal + delta)

        offers = [o for o in self.nmi.outcome_space if o not in self.proposed_offers]
        if not offers:
            return None


        opp_type, ubi, _ = self.classify_opponent()
        if opp_type == "boulware" and ubi > 0:
            late_round = t > 1.0 - (0.5 / ubi)
            if late_round:
                self.m = min(self.m, 0.3)
                if self.best_received_offer is not None and self.best_received_utility >= self.m:
                    self.proposed_offers.add(self.best_received_offer)
                    return self.best_received_offer

        acceptable = [o for o in offers if self.ufun(o) >= aspiration]

        if acceptable:

            candidates = [o for o in acceptable if lower <= self.ufun(o) <= upper]
            if candidates:
                offer = min(candidates, key=lambda o: self.ufun(o) - goal)
            else:
                offer = min(acceptable, key=lambda o: self.ufun(o) - aspiration)
        else:
            offer = max(offers, key=self.ufun)

        if self.best_received_offer is not None and self.best_received_utility > self.ufun(offer):
            offer = self.best_received_offer

        self.proposed_offers.add(offer)
        return offer


    def respond(self, state, source=None):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        self.opponent_offers_history.append((state.relative_time, offer))


        offer_u = self.ufun(offer)
        if offer_u > self.best_received_utility:
            self.best_received_offer = offer
            self.best_received_utility = offer_u

        next_offer, sigma = self.opponent_strategy(state)

        predicted_utility = self.ufun(next_offer)
        self.opponent_ufun_predictions_history.append((state.relative_time, predicted_utility))

        sigma_scalar = float(np.mean(sigma))
        threshold = max(self.aspiration(state), predicted_utility - sigma_scalar)

        if offer_u >= threshold:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


    def opponent_strategy(self, state):
        if len(self.opponent_offers_history) < 2:
            last_offer = (
                np.array(self.opponent_offers_history[-1][1])
                if self.opponent_offers_history
                else np.zeros(len(self.nmi.issues))
            )
            return last_offer, np.zeros_like(last_offer)

        times = np.array([t for t, o in self.opponent_offers_history]).reshape(-1, 1)
        offers = np.array([o for t, o in self.opponent_offers_history])
        n_issues = offers.shape[1]

        predicted_offer = np.zeros(n_issues)
        sigma = np.zeros(n_issues)

        kernel = RBF(length_scale=0.05) + WhiteKernel(noise_level=1e-2)

        for i in range(n_issues):
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            gp.fit(times, offers[:, i])
            pred, std = gp.predict(np.array([[state.relative_time]]), return_std=True)
            predicted_offer[i] = pred[0]
            sigma[i] = std[0]

        return predicted_offer, sigma