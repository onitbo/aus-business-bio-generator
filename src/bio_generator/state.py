from __future__ import annotations

"""Typed state for the LangGraph bio generation workflow."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class BioState(TypedDict, total=False):
    """State that flows through the LangGraph StateGraph."""
    # Input
    name: str
    address: str
    region: str

    # Research
    places_data: Dict[str, Any]
    website_data: Dict[str, Any]
    serper_data: Dict[str, Any]
    research_context: str
    sources: List[Dict[str, Any]]

    # Sufficiency
    sufficient: bool

    # Bio drafting / revision
    description: str
    confidence: float
    review_passed: bool
    review_issues: List[str]
    review_suggestions: List[str]
    iteration: int

    # Token tracking
    token_usage: Dict[str, Dict[str, int]]

    # Output
    slug: str
    description_id: str
    errors: List[str]
    status: str
