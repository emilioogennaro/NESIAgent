from negmas.sao import SAONegotiator, SAOResponse, ResponseType, SAOState
from components.acceptance import AcceptanceStrategy, StaticThresholdAcceptance
from components.bidding import BiddingStrategy, RandomAboveThresholdBidding
from components.opponent_model import OpponentModel, NoOpponentModel

class Group37_Negotiator(SAONegotiator):
    """
    The main negotiating agent utilizing a composition architecture.
    """
    
    def __init__(self, *args, 
                 bidding_strategy: BiddingStrategy = None, 
                 acceptance_strategy: AcceptanceStrategy = None, 
                 opponent_model: OpponentModel = None, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Instantiate default components if none are explicitly injected
        self.bidding_strategy = bidding_strategy or RandomAboveThresholdBidding(threshold=0.9)
        self.acceptance_strategy = acceptance_strategy or StaticThresholdAcceptance(threshold=0.8)
        self.opponent_model = opponent_model or NoOpponentModel()

    def __call__(self, state: SAOState) -> SAOResponse:
        """The main turn cycle called by the NegMAS framework."""
        offer = state.current_offer

        if offer is not None:
            # 1. Update our knowledge of the opponent
            self.opponent_model.update(offer, state)
            
            # 2. Ask the acceptance component if we should agree
            if self.acceptance_strategy.evaluate(offer, state, self.ufun):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # 3. If no agreement, ask the bidding component for a counter-offer
        my_proposal = self.bidding_strategy.generate(state, self.ufun, self.nmi)
        return SAOResponse(ResponseType.REJECT_OFFER, my_proposal)