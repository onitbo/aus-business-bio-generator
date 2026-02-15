"""Single business bio generator entry point."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bio_generator.config import Config
from bio_generator.graph import build_graph


async def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python main.py <name> <address> <region>")
        print("Example: python main.py 'Elite Euro' '9A Berriman Drive Wangara' 'WA'")
        sys.exit(1)

    name, address, region = sys.argv[1], sys.argv[2], sys.argv[3]

    config = Config.from_env()
    app = build_graph(config)

    print("🚀 Generating bio for: %s" % name)
    final_state = await app.ainvoke({
        "name": name,
        "address": address,
        "region": region,
    })

    print("\n📝 Description:")
    print(final_state.get("description", ""))
    print("\nConfidence: %s" % final_state.get("confidence", 0.0))


if __name__ == "__main__":
    asyncio.run(main())
