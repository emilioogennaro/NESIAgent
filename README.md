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
