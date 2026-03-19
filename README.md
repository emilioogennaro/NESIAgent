# NESI Negotiation Agent

This repository implements **NESI**, a negotiation agent developed for CSE3210
Collaborative Artificial Intelegence course.  NESI is built on top of the
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
