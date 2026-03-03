import os
import sys
# type: ignore

# Ensure Python can find our 'agents' and 'components' folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from negmas.sao import SAOMechanism
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao.negotiators import RandomNegotiator

# Import your custom agent
from agent.Group37_Negotiator import Group37_Negotiator

def run_single_negotiation():
    print("--- Starting 1v1 Negotiation Sandbox ---")

    # 1. Define the Negotiation Domain (The Issues)
    # We create two issues: Price (0-100) and Delivery Time (0-30 days)
    issues = [
        make_issue(name="Price", values=101),
        make_issue(name="Delivery_Days", values=31)
    ]

    # 2. Generate Random Linear Additive Utility Functions for both agents
    ufun_a = LUFun.random(issues=issues, reserved_value=(0.1, 0.3))
    ufun_b = LUFun.random(issues=issues, reserved_value=(0.1, 0.3))

    # 3. Create the Negotiation Session (The Mechanism)
    # We set a limit of 100 steps. No discounting is applied as per assignment rules.
    session = SAOMechanism(issues=issues, n_steps=100)

    # 4. Initialize the Agents
    # We inject your agent, and a baseline RandomNegotiator provided by NegMAS
    agent_a = Group37_Negotiator(name="Group37_Agent")
    agent_b = RandomNegotiator(name="Baseline_Random")

    # 5. Add Agents to the Session with their respective utilities
    session.add(agent_a, ufun=ufun_a)
    session.add(agent_b, ufun=ufun_b)

    # 6. Run the Simulation
    state = session.run()

    # 7. Output the Results
    print("\n--- Negotiation Finished ---")
    if state.agreement:
        print(f"Agreement Reached: {state.agreement}")
        print(f"Utility for {agent_a.name}: {ufun_a(state.agreement):.3f}")
        print(f"Utility for {agent_b.name}: {ufun_b(state.agreement):.3f}")
    else:
        print("Negotiation Failed (Walkaway or Timeout).")
        print(f"Fallback Utility for {agent_a.name}: {ufun_a.reserved_value:.3f}")
        print(f"Fallback Utility for {agent_b.name}: {ufun_b.reserved_value:.3f}")

if __name__ == "__main__":
    run_single_negotiation()