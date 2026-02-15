from __future__ import annotations

"""LangGraph node functions for the bio generation workflow."""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import Config
from .llm import LLMClient
from .prompts import (
    DRAFT_SYSTEM,
    DRAFT_USER,
    REVIEW_SYSTEM,
    REVIEW_USER,
    REVISE_SYSTEM,
    REVISE_USER,
)
from .research import fetch_website_content, search_google_places, slugify
from .state import BioState

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator


# Module-level config and LLM client, set before graph execution
_config = None  # type: Any
_llm = None  # type: Any


def set_config(config: Config) -> None:
    global _config, _llm
    _config = config
    _llm = LLMClient(
        api_key=config.anthropic_api_key,
        model=config.model_name,
        temperature=config.model_temperature,
    )


def _parse_review_json(text: str) -> Dict[str, Any]:
    """Robustly parse review JSON from LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "confidence": 0.5,
        "issues": ["Could not parse review response"],
        "suggestions": [],
        "has_markdown": False,
        "factual_accuracy": 0.5,
        "readability": 0.5,
        "passes": False,
    }


def _strip_markdown(text: str) -> str:
    """Remove any markdown formatting from text."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text.strip()


@traceable(name="ingest_input")
def ingest_input(state: BioState) -> Dict[str, Any]:
    """Initialize state with input data."""
    print("  📥 Ingesting input for %s..." % state["name"])
    return {
        "slug": slugify(state["name"]),
        "description_id": str(uuid.uuid4()),
        "iteration": 0,
        "confidence": 0.0,
        "review_passed": False,
        "errors": [],
        "token_usage": {},
        "sources": [],
        "status": "processing",
    }


@traceable(name="research_places")
async def research_places(state: BioState) -> Dict[str, Any]:
    """Search Google Places for business info."""
    print("  📡 Searching Google Places for %s..." % state["name"])
    try:
        places_data = await search_google_places(
            state["name"], state["address"], state["region"],
            _config.google_places_api_key,
        )
    except Exception as e:
        print("  ⚠️  Places search failed: %s" % e)
        return {
            "places_data": {"found": False},
            "errors": state.get("errors", []) + ["Places search failed: %s" % e],
        }

    sources = [{
        "name": "Google Places",
        "type": "google_places",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "raw_excerpt_char_count": len(places_data.get("raw_excerpt", "")),
        "notes": "place_id: %s" % (places_data.get("place_id") or "not_found"),
    }]

    return {"places_data": places_data, "sources": sources}


@traceable(name="research_website_from_places")
async def research_website_from_places(state: BioState) -> Dict[str, Any]:
    """Fetch content from the website found via Google Places."""
    places_data = state.get("places_data", {})
    website_url = places_data.get("website")

    if not website_url:
        print("  ℹ️  No website found, skipping website research")
        return {"website_data": {"url": None, "content": "", "fetched": False}}

    print("  🌐 Fetching website: %s" % website_url)
    website_data = await fetch_website_content(website_url)

    sources = list(state.get("sources", []))
    if website_data.get("fetched"):
        sources.append({
            "name": website_url,
            "type": "website",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "raw_excerpt_char_count": website_data.get("char_count", 0),
            "notes": "Extracted from business website",
        })

    # Build combined research context
    context_parts = []  # type: List[str]
    if places_data.get("raw_excerpt"):
        context_parts.append("=== Google Places Data ===\n" + places_data["raw_excerpt"])
    if website_data.get("content"):
        context_parts.append("=== Website Content ===\n" + website_data["content"])

    research_context = "\n\n".join(context_parts) if context_parts else "Business: %s\nAddress: %s, %s" % (
        state["name"], state["address"], state["region"]
    )

    return {
        "website_data": website_data,
        "sources": sources,
        "research_context": research_context,
    }


@traceable(name="decide_sufficiency")
def decide_sufficiency(state: BioState) -> Dict[str, Any]:
    """Decide if we have enough research data to write a bio."""
    context = state.get("research_context", "")
    places_data = state.get("places_data", {})

    sufficient = bool(places_data.get("found", False) and len(context) > 100)
    if not sufficient:
        print("  ⚠️  Insufficient research data")
    else:
        print("  ✅ Sufficient research data collected")

    return {"sufficient": sufficient}


@traceable(name="draft_bio")
def draft_bio(state: BioState) -> Dict[str, Any]:
    """Draft the business bio using LLM."""
    print("  ✍️  Drafting bio...")

    if not state.get("sufficient", True):
        return {
            "description": "unable to create bio due to lack of information found online.",
            "confidence": 0.0,
            "status": "fallback",
        }

    draft = _llm.generate(
        DRAFT_SYSTEM,
        DRAFT_USER.format(
            name=state["name"],
            address=state["address"],
            region=state["region"],
            context=state.get("research_context", ""),
        ),
        step="draft",
    )
    draft = _strip_markdown(draft)

    return {
        "description": draft,
        "iteration": 1,
        "token_usage": _llm.usage.by_step.copy(),
    }


@traceable(name="review_bio")
def review_bio(state: BioState) -> Dict[str, Any]:
    """Review the current bio draft."""
    iteration = state.get("iteration", 1)
    print("  🔍 Review iteration %d..." % iteration)

    review_text = _llm.generate(
        REVIEW_SYSTEM,
        REVIEW_USER.format(
            name=state["name"],
            address=state["address"],
            region=state["region"],
            context=state.get("research_context", ""),
            description=state.get("description", ""),
        ),
        step="review",
    )

    review = _parse_review_json(review_text)
    confidence = review.get("confidence", 0.5)
    passes = review.get("passes", False)

    if passes and confidence >= _config.target_confidence:
        print("  ✅ Passed review (confidence: %s)" % confidence)
    else:
        print("  ❌ Review: confidence=%s, passes=%s" % (confidence, passes))

    return {
        "confidence": confidence,
        "review_passed": bool(passes and confidence >= _config.target_confidence),
        "review_issues": review.get("issues", []),
        "review_suggestions": review.get("suggestions", []),
        "token_usage": _llm.usage.by_step.copy(),
    }


@traceable(name="revise_bio")
def revise_bio(state: BioState) -> Dict[str, Any]:
    """Revise the bio based on review feedback."""
    iteration = state.get("iteration", 1)
    print("  🔄 Revising (iteration %d)..." % iteration)

    issues = "; ".join(state.get("review_issues", []))
    suggestions = "; ".join(state.get("review_suggestions", []))

    revised = _llm.generate(
        REVISE_SYSTEM,
        REVISE_USER.format(
            name=state["name"],
            address=state["address"],
            region=state["region"],
            context=state.get("research_context", ""),
            description=state.get("description", ""),
            issues=issues,
            suggestions=suggestions,
        ),
        step="revise",
    )
    revised = _strip_markdown(revised)

    return {
        "description": revised,
        "iteration": iteration + 1,
        "token_usage": _llm.usage.by_step.copy(),
    }


@traceable(name="finalize_and_validate")
def finalize_and_validate(state: BioState) -> Dict[str, Any]:
    """Finalize the bio, applying fallback if needed."""
    confidence = state.get("confidence", 0.0)
    review_passed = state.get("review_passed", False)
    iteration = state.get("iteration", 0)
    description = state.get("description", "")

    # If max iterations reached and still failing, use fallback
    if iteration >= _config.max_iterations and not review_passed:
        print("  ⚠️  Max iterations reached with failing review - using fallback")
        description = "unable to create bio due to lack of information found online."
        confidence = 0.0

    # Final markdown strip
    description = _strip_markdown(description)

    return {
        "description": description,
        "confidence": confidence,
        "status": "ok" if confidence >= _config.target_confidence else "low_confidence",
        "token_usage": _llm.usage.by_step.copy(),
    }


@traceable(name="write_outputs")
def write_outputs(state: BioState) -> Dict[str, Any]:
    """Write the bio and metadata to files."""
    output_dir = Path(_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = state["slug"]

    # Write bio
    bio_path = output_dir / ("%s.md" % slug)
    bio_path.write_text(state.get("description", ""))

    # Write metadata
    metadata = {
        "description_id": state.get("description_id", ""),
        "business_input": {
            "name": state["name"],
            "address_no_postcode": state["address"],
            "region": state["region"],
        },
        "run": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "langsmith_project": _config.langchain_project if _config.langchain_tracing else None,
            "langsmith_trace_url": None,
            "thread_id": "bio-%s-%s" % (slug, state.get("description_id", "")[:8]),
        },
        "sources": state.get("sources", []),
        "token_usage": _llm.usage.to_dict() if _llm else {},
        "confidence": state.get("confidence", 0.0),
        "status": state.get("status", "unknown"),
        "errors": state.get("errors", []),
        "model_info": {
            "provider": _config.model_provider,
            "model": _config.model_name,
            "temperature": _config.model_temperature,
        },
    }

    meta_path = output_dir / ("%s_description_meta.json" % slug)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    print("  💾 Saved: %s + %s" % (bio_path, meta_path))

    return {"status": metadata["status"]}
