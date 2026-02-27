# NESI Negotiation Agent

This repository implements **NESI**, a negotiation agent developed for CSE3210
Human–Computer Interaction course.  NESI is built on top of the
[NegMAS](https://github.com/negmas-org/negmas) multi‑agent negotiation framework.

## Repository layout

- `src/nesi_agent`: Python package containing the NESI agent source code.
- `tests`: Unit test suite (pytest).
- `pyproject.toml`: Project manifest with metadata and dependency declarations.

## Setup & installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

After activation the `negmas` dependency will be available and the package can be imported:

```python
from nesi_agent.agent import NESIAgent
```

Run the test suite with `pytest`.

## Development notes

Implementation of the negotiation strategy resides in
`src/nesi_agent/agent.py`.  The class `NESIAgent` provides skeleton code for
`choose_action()` and `handle_event()`; concrete logic should be added there.

IVAN WRITTEN WISDOM

so guys like we have the Group37_Negotiator, it accepts the abstract classes AcceptanceStrategy and BiddginStrategy. We can implement many such strategies and just plug them in when creating an instance. 

There is also some code for running 1v1 and tournament. We touch that way in the future. 

Gemini wrote all of this of course but i told him that i want modular plug and play architecture so i also did some shit. 
