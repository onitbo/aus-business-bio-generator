from __future__ import annotations

"""LangGraph StateGraph definition for the bio generation workflow."""

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from .config import Config
from .nodes import (
    decide_sufficiency,
    draft_bio,
    finalize_and_validate,
    ingest_input,
    research_places,
    research_serper_node,
    research_website_from_places,
    review_bio,
    revise_bio,
    set_config,
    write_outputs,
)
from .state import BioState


def _review_router(state: BioState) -> str:
    """Route after review: finalize if passed, revise if not (and under max iterations)."""
    if state.get("review_passed", False):
        return "finalize_and_validate"

    config = _get_config()
    if state.get("iteration", 0) >= config.max_iterations:
        return "finalize_and_validate"

    return "revise_bio"


_config_ref = None  # type: Any


def _get_config() -> Config:
    return _config_ref


def build_graph(config: Config) -> StateGraph:
    """Build and compile the bio generation StateGraph."""
    global _config_ref
    _config_ref = config
    set_config(config)

    graph = StateGraph(BioState)

    # Add nodes
    graph.add_node("ingest_input", ingest_input)
    graph.add_node("research_places", research_places)
    graph.add_node("research_website_from_places", research_website_from_places)
    graph.add_node("research_serper", research_serper_node)
    graph.add_node("decide_sufficiency", decide_sufficiency)
    graph.add_node("draft_bio", draft_bio)
    graph.add_node("review_bio", review_bio)
    graph.add_node("revise_bio", revise_bio)
    graph.add_node("finalize_and_validate", finalize_and_validate)
    graph.add_node("write_outputs", write_outputs)

    # Set entry point
    graph.set_entry_point("ingest_input")

    # Add edges
    graph.add_edge("ingest_input", "research_places")
    graph.add_edge("research_places", "research_website_from_places")
    graph.add_edge("research_website_from_places", "research_serper")
    graph.add_edge("research_serper", "decide_sufficiency")
    graph.add_edge("decide_sufficiency", "draft_bio")
    graph.add_edge("draft_bio", "review_bio")

    # Conditional: review_bio → finalize or revise
    graph.add_conditional_edges(
        "review_bio",
        _review_router,
        {
            "finalize_and_validate": "finalize_and_validate",
            "revise_bio": "revise_bio",
        },
    )

    # revise_bio loops back to review_bio
    graph.add_edge("revise_bio", "review_bio")

    # finalize → write → END
    graph.add_edge("finalize_and_validate", "write_outputs")
    graph.add_edge("write_outputs", END)

    return graph.compile()
