from __future__ import annotations

"""Core bio generation pipeline: research -> draft -> review -> revise."""

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
from .research import gather_research, slugify


def _parse_review_json(text):
    """Robustly parse review JSON from LLM response."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: return a low-confidence review
    return {
        "confidence": 0.5,
        "issues": ["Could not parse review response"],
        "suggestions": [],
        "has_markdown": False,
        "factual_accuracy": 0.5,
        "readability": 0.5,
        "passes": False,
    }


def _strip_markdown(text):
    """Remove any markdown formatting from text."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text.strip()


async def generate_bio(name, address, region, config):
    """Generate a business bio with research, drafting, review, and revision."""
    started_at = datetime.now(timezone.utc)
    slug = slugify(name)
    description_id = str(uuid.uuid4())
    errors = []

    llm = LLMClient(
        api_key=config.anthropic_api_key,
        model=config.model_name,
        temperature=config.model_temperature,
    )

    # Step 1: Research
    print("  Researching {}...".format(name))
    try:
        research = await gather_research(name, address, region, config.google_places_api_key)
    except Exception as e:
        errors.append("Research failed: {}".format(e))
        research = {
            "context": "Business: {}\nAddress: {}, {}".format(name, address, region),
            "sources": [],
            "places_data": {},
            "website_data": {},
        }

    context = research["context"]
    sources = research["sources"]
    place_id = research.get("places_data", {}).get("place_id")

    # Step 2: Draft
    print("  Drafting bio...")
    draft = llm.generate(
        DRAFT_SYSTEM,
        DRAFT_USER.format(name=name, address=address, region=region, context=context),
        step="draft",
    )
    draft = _strip_markdown(draft)

    # Step 3: Review + Revise loop
    description = draft
    confidence = 0.0

    for iteration in range(config.max_iterations):
        print("  Review iteration {}...".format(iteration + 1))
        review_text = llm.generate(
            REVIEW_SYSTEM,
            REVIEW_USER.format(
                name=name, address=address, region=region,
                context=context, description=description,
            ),
            step="review",
        )

        review = _parse_review_json(review_text)
        confidence = review.get("confidence", 0.5)

        if confidence >= config.target_confidence and review.get("passes", False):
            print("  Passed review (confidence: {})".format(confidence))
            break

        if iteration < config.max_iterations - 1:
            print("  Revising (confidence: {})...".format(confidence))
            issues = "; ".join(review.get("issues", []))
            suggestions = "; ".join(review.get("suggestions", []))
            description = llm.generate(
                REVISE_SYSTEM,
                REVISE_USER.format(
                    name=name, address=address, region=region,
                    context=context, description=description,
                    issues=issues, suggestions=suggestions,
                ),
                step="revise",
            )
            description = _strip_markdown(description)
    else:
        print("  Max iterations reached (confidence: {})".format(confidence))

    description = _strip_markdown(description)
    finished_at = datetime.now(timezone.utc)

    metadata = {
        "description_id": description_id,
        "business_input": {
            "name": name,
            "address_no_postcode": address,
            "region": region,
        },
        "run": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "langsmith_project": config.langchain_project if config.langchain_tracing else None,
            "langsmith_trace_url": None,
            "thread_id": "bio-{}-{}".format(slug, description_id[:8]),
        },
        "sources": _serialize_sources(sources),
        "token_usage": llm.usage.to_dict(),
        "confidence": confidence,
        "status": "ok" if not errors else "partial",
        "errors": errors,
        "model_info": {
            "provider": config.model_provider,
            "model": config.model_name,
            "temperature": config.model_temperature,
        },
    }

    return {
        "description": description,
        "metadata": metadata,
        "slug": slug,
    }


def _serialize_sources(sources):
    """Ensure all source values are JSON-serializable (handles datetime)."""
    serialized = []
    for source in sources:
        s = {}
        for k, v in source.items():
            if isinstance(v, datetime):
                s[k] = v.isoformat()
            else:
                s[k] = v
        serialized.append(s)
    return serialized


async def generate_and_save(name, address, region, config):
    """Generate bio and save to output files."""
    result = await generate_bio(name, address, region, config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = result["slug"]

    bio_path = output_dir / "{}.md".format(slug)
    bio_path.write_text(result["description"])

    meta_path = output_dir / "{}_description_meta.json".format(slug)
    meta_path.write_text(json.dumps(result["metadata"], indent=2, default=str))

    print("  Saved: {} + {}".format(bio_path, meta_path))
    return result
