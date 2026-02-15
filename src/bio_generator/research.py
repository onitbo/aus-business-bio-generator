from __future__ import annotations

"""Research step: gather data from Google Places and business website."""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx


def slugify(name):
    """Convert business name to URL-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


async def search_google_places(name, address, region, api_key):
    """Search Google Places for business info. Returns place data + place_id."""
    query = "{} {} {} Australia".format(name, address, region)
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,
        "inputtype": "textquery",
        "fields": "place_id,name,formatted_address,rating,user_ratings_total,types,business_status,opening_hours",
        "key": api_key,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        return {"place_id": None, "found": False, "raw": data}

    place = candidates[0]
    place_id = place.get("place_id")

    # Get place details for richer info
    details = {}
    if place_id:
        details = await _get_place_details(place_id, api_key)

    return {
        "place_id": place_id,
        "found": True,
        "name": place.get("name", name),
        "address": place.get("formatted_address", address),
        "rating": place.get("rating"),
        "review_count": place.get("user_ratings_total"),
        "website": details.get("website") or place.get("website"),
        "types": place.get("types", []),
        "business_status": place.get("business_status"),
        "phone": details.get("formatted_phone_number"),
        "reviews": details.get("reviews", [])[:5],
        "raw_excerpt": _summarise_place(place, details),
    }


async def _get_place_details(place_id, api_key):
    """Fetch detailed place information."""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "website,formatted_phone_number,reviews,editorial_summary,url",
        "key": api_key,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    return data.get("result", {})


def _summarise_place(place, details):
    """Build a text summary from place data for the LLM."""
    parts = []
    parts.append("Business: {}".format(place.get("name", "Unknown")))
    parts.append("Address: {}".format(place.get("formatted_address", "Unknown")))
    if place.get("rating"):
        parts.append("Rating: {}/5 ({} reviews)".format(place["rating"], place.get("user_ratings_total", 0)))
    if place.get("types"):
        parts.append("Categories: {}".format(", ".join(place["types"])))
    if place.get("business_status"):
        parts.append("Status: {}".format(place["business_status"]))
    if details.get("editorial_summary", {}).get("overview"):
        parts.append("Summary: {}".format(details["editorial_summary"]["overview"]))
    for i, rev in enumerate(details.get("reviews", [])[:3], 1):
        text = rev.get("text", "")[:200]
        if text:
            parts.append("Review {}: {}".format(i, text))
    return "\n".join(parts)


async def fetch_website_content(url, max_chars=3000):
    """Fetch and extract text from a business website."""
    if not url:
        return {"url": None, "content": "", "fetched": False}

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "AusBioGen/1.0"})
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        return {"url": url, "content": "", "fetched": False, "error": str(e)}

    text = _html_to_text(html)
    return {
        "url": url,
        "content": text[:max_chars],
        "char_count": min(len(text), max_chars),
        "fetched": True,
    }


def _html_to_text(html):
    """Very basic HTML to text - strip tags, collapse whitespace."""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
        text = text.replace(entity, char)
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def gather_research(name, address, region, api_key):
    """Run all research steps and return combined data."""
    retrieved_at = datetime.now(timezone.utc).isoformat()

    places_data = await search_google_places(name, address, region, api_key)

    website_url = places_data.get("website")
    website_data = await fetch_website_content(website_url) if website_url else {
        "url": None, "content": "", "fetched": False
    }

    sources = []
    sources.append({
        "name": "Google Places",
        "type": "google_places",
        "retrieved_at": retrieved_at,
        "raw_excerpt_char_count": len(places_data.get("raw_excerpt", "")),
        "notes": "place_id: {}".format(places_data.get("place_id") or "not_found"),
    })

    if website_data.get("fetched"):
        sources.append({
            "name": website_url,
            "type": "website",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "raw_excerpt_char_count": website_data.get("char_count", 0),
            "notes": "Extracted from business website",
        })

    context_parts = []
    if places_data.get("raw_excerpt"):
        context_parts.append("=== Google Places Data ===\n" + places_data["raw_excerpt"])
    if website_data.get("content"):
        context_parts.append("=== Website Content ===\n" + website_data["content"])

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
        "places_data": places_data,
        "website_data": website_data,
    }
