"""Unit tests for the NegMas agent package."""

import pytest

from nesi_agent.agent import NESIAgent


def test_agent_initialization():
    agent = NESIAgent()
    assert agent is not None


# further tests can be added here
