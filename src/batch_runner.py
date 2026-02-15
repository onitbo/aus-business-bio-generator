"""Batch runner for generating bios for multiple Australian businesses."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bio_generator.config import Config
from bio_generator.graph import build_graph

# Target businesses - exact addresses from spec
BUSINESSES = [
    {"name": "Scobies Service Centre", "address": "155A LEARMONTH STREET Ballarat", "region": "VIC"},
    {"name": "Elite Euro", "address": "9A Berriman Drive Wangara", "region": "WA"},
    {"name": "Launceston BMW", "address": "143 Invermay Road Invermay", "region": "TAS"},
    {"name": "Daimler Trucks Toowoomba", "address": "351 Taylor Street Toowoomba", "region": "QLD"},
    {"name": "CJD Equipment Darwin", "address": "1 Toupein Rd Yarrawonga", "region": "NT"},
]


async def main() -> None:
    config = Config.from_env()
    print("🚀 Australian Business Bio Generator (LangGraph)")
    print("   Model: %s/%s" % (config.model_provider, config.model_name))
    print("   Target confidence: %s" % config.target_confidence)
    print("   Output dir: %s" % config.output_dir)
    print()

    app = build_graph(config)

    results = []
    for biz in BUSINESSES:
        print("📋 Processing: %s" % biz["name"])
        try:
            initial_state = {
                "name": biz["name"],
                "address": biz["address"],
                "region": biz["region"],
            }
            final_state = await app.ainvoke(initial_state)
            conf = final_state.get("confidence", 0.0)
            status = final_state.get("status", "unknown")
            print("  ✅ Done — confidence: %s, status: %s" % (conf, status))
            results.append(final_state)
        except Exception as e:
            print("  ❌ Failed: %s" % e)
            import traceback
            traceback.print_exc()
        print()

    # Summary
    print("=" * 60)
    print("📊 BATCH SUMMARY")
    print("=" * 60)
    for r in results:
        conf = r.get("confidence", 0.0)
        name = r.get("name", "?")
        status_icon = "✅" if conf >= config.target_confidence else "⚠️"
        print("  %s %s: confidence=%s" % (status_icon, name, conf))
    print("\n  Businesses processed: %d/%d" % (len(results), len(BUSINESSES)))


if __name__ == "__main__":
    asyncio.run(main())
