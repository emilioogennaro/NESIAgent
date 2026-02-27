import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from negmas.sao import SAOMechanism
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao.negotiators import RandomNegotiator, AspirationNegotiator

from agent.Group37_Negotiator import Group37_Negotiator

def run_tournament(n_matches=10):
    print(f"--- Running Tournament ({n_matches} matches) ---")
    
    results = []

    for match_id in range(n_matches):
        # Generate a completely new random domain and utilities for every match
        issues = [
            make_issue(name=f"Issue_A_{match_id}", values=50),
            make_issue(name=f"Issue_B_{match_id}", values=50)
        ]
        
        ufun_groupN = LUFun.random(issues=issues, reserved_value=(0.0, 0.2))
        ufun_opponent = LUFun.random(issues=issues, reserved_value=(0.0, 0.2))

        session = SAOMechanism(issues=issues, n_steps=100)
        
        # Alternate opponents for variety (e.g., Random vs Aspiration)
        opponent = RandomNegotiator(name="Random_Opponent") if match_id % 2 == 0 else AspirationNegotiator(name="Aspiration_Opponent")
        group_n_agent = Group37_Negotiator(name="Group37")

        session.add(group_n_agent, ufun=ufun_groupN)
        session.add(opponent, ufun=ufun_opponent)
        
        state = session.run()
        
        # Collect match data
        match_data = {
            "Match_ID": match_id,
            "Opponent": opponent.name,
            "Agreement_Reached": state.agreement is not None,
            "Steps_Taken": state.step,
            "GroupN_Utility": ufun_groupN(state.agreement) if state.agreement else ufun_groupN.reserved_value,
            "Opponent_Utility": ufun_opponent(state.agreement) if state.agreement else ufun_opponent.reserved_value,
        }
        results.append(match_data)
        print(f"Match {match_id} completed. Agreement: {match_data['Agreement_Reached']}")

    # Convert results to a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv("tournament_results.csv", index=False)
    
    print("\n--- Tournament Summary ---")
    print(df.groupby("Opponent")["GroupN_Utility"].mean())
    print("\nResults saved to tournament_results.csv")

if __name__ == "__main__":
    run_tournament(n_matches=10)