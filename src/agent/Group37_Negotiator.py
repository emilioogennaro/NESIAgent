# type: ignore
from typing import Any
from negmas.sao import SAONegotiator, SAOResponse, ResponseType, SAOState
from components.acceptance import HybridAcceptance
from components.bidding import OpponentAwareBidding
from components.opponent_model import FrequencyAnalysisModel

class Group37_Negotiator(SAONegotiator):
    """
    The main negotiating agent utilizing a composition architecture.
    """
    
    def __init__(self, *args,
             bidding_strategy=None,
             acceptance_strategy=None,
             opponent_model=None,
             **kwargs):
        super().__init__(*args, **kwargs)

        self.opponent_model = opponent_model or FrequencyAnalysisModel()
        self.bidding_strategy = bidding_strategy or OpponentAwareBidding(opponent_model=self.opponent_model)
        self.acceptance_strategy = acceptance_strategy or HybridAcceptance()

    def __call__(self, state: SAOState, *args: Any, **kwargs: Any) -> SAOResponse:
        """The main turn cycle called by the NegMAS framework."""
        offer = state.current_offer

        if offer is not None:
            self.opponent_model.update(offer, state)
            
            if self.acceptance_strategy.evaluate(offer, state, self.ufun):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        my_proposal = self.bidding_strategy.generate(state, self.ufun, self.nmi)
        return SAOResponse(ResponseType.REJECT_OFFER, my_proposal)