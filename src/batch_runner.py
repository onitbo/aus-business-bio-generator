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
    {"name": "A Team Mechanical", "address": "278 Long Plains Road Exeter", "region": "TAS"},
    {"name": "Wayne Wood Auto Electrical", "address": "320 Rifle Butts Road Katandra", "region": "VIC"},
    {"name": "Rival Prestige Panel & Paint", "address": "480 Duncan Road Nicholson", "region": "VIC"},
    {"name": "JB Mechanical and Suspension", "address": "4-6 Gedge Street Ingham", "region": "QLD"},
    {"name": "Bond Street Auto Electrics", "address": "30-36 Bond Street Sale", "region": "VIC"},
    {"name": "Valley Automotive", "address": "411 Main Road Coromandel Valley", "region": "SA"},
    {"name": "Jagtrack Pty Ltd", "address": "484 Victoria Street Wetherill Park", "region": "NSW"},
    {"name": "KJ's Service & Repairs", "address": "351 Birt Road Corndale", "region": "QLD"},
    {"name": "Mick's Auto Repairs", "address": "437 Yarrie Lake Road Narrabri", "region": "NSW"},
    {"name": "SEA Heavy Diesel", "address": "6, 15 Hill Street Esperance", "region": "WA"},
    {"name": "Auto-Fab", "address": "25 Keysborough Close Keysborough", "region": "VIC"},
    {"name": "J&K Mechanical", "address": "21 Beafield Road Para Hills West", "region": "SA"},
    {"name": "Atkins mechanical service", "address": "44 Richardson Street Brookton", "region": "WA"},
    {"name": "Tyrepower Gisborne", "address": "7 Hamilton Street Gisborne", "region": "VIC"},
    {"name": "Tyrepower Enoggera", "address": "62 Pickering Street Enoggera", "region": "QLD"},
    {"name": "King Island Motors", "address": "1 Netherby Road Currie", "region": "TAS"},
    {"name": "Outback AG Services", "address": "10 Foley Court Mount Tarcoola", "region": "WA"},
    {"name": "Coastal Mechanical Repairs", "address": "28 vestam Court Carrara", "region": "QLD"},
    {"name": "Adaminaby Automotive Repairs", "address": "5172 Snowy Mountains Highway Adaminaby", "region": "NSW"},
    {"name": "T.A.S Automotive", "address": "7 Civil Court Harlaxton", "region": "QLD"},
    {"name": "Mobile Air Compressor Services", "address": "9 Verbena Terrace Epsom", "region": "VIC"},
    {"name": "Southwest Automotive Services", "address": "460a Grossmans Road Bellbrae", "region": "VIC"},
    {"name": "EMS Truck & Plant Repairs", "address": "25 Manilla Street Manilla", "region": "NSW"},
    {"name": "Ice Auto Body", "address": "8 George Street Clyde", "region": "NSW"},
    {"name": "IC Logistics", "address": "46 Helen Street Warilla", "region": "NSW"},
    {"name": "AJM Automotive", "address": "1/112 Montague Street North Wollongong", "region": "NSW"},
    {"name": "Hardman Mechanical Services", "address": "3/15 Shearer Drive Seaford", "region": "SA"},
    {"name": "All Automotive Servicing", "address": "11 Egret Walk Rowville", "region": "VIC"},
    {"name": "Envis Digital Innovations", "address": "11 Lugarno Parade Lugarno", "region": "NSW"},
    {"name": "R & R Motors", "address": "79 Douglas Street Thursday Island", "region": "QLD"},
    {"name": "Stewarts Automotive Services", "address": "11 Sanger Street Corowa", "region": "NSW"},
    {"name": "TM Automotive", "address": "23 Boscobel Road Londonderry", "region": "NSW"},
    {"name": "Eastern Creek Mechanical Repairs", "address": "32 Holbeche Road Arndell Park", "region": "NSW"},
    {"name": "L. Sandstrom Installations", "address": "8 Fairlight Way Culburra Beach", "region": "NSW"},
    {"name": "ASA Engineering", "address": "23 Wellington Terrace Fullarton", "region": "SA"},
    {"name": "HeavyIron Group", "address": "15 Barfield Crescent Edinburgh North", "region": "SA"},
    {"name": "Outback Mechanical Services", "address": "18 Margaret Avenue Stirling North", "region": "SA"},
    {"name": "SA Truck Curtain Repairs", "address": "2 Capelli Road Wingfield", "region": "SA"},
    {"name": "Chris's Mechanical Services", "address": "12 Wilpena Road Hawker", "region": "SA"},
    {"name": "BMP Installation", "address": "75 Liston Street Glen Iris", "region": "VIC"},
    {"name": "Enterprise Auto Repairs", "address": "3/47 Enterprise Street Kunda Park", "region": "QLD"},
    {"name": "Les Mechanical", "address": "15 Pharlap Street Russell Island", "region": "QLD"},
    {"name": "Hughes Mechanical Yorke Peninsula", "address": "60 Minlaton Road Yorketown", "region": "SA"},
    {"name": "Hills Tractor Fix", "address": "3 Stephens Close Endeavour Hills", "region": "VIC"},
    {"name": "Specialised Electrical Testing Company", "address": "623 Orrong Road Toorak", "region": "VIC"},
    {"name": "RB Mechanical Mobile", "address": "9 Boondoora Drive Calliope", "region": "QLD"},
    {"name": "LMH Repairers", "address": "3 Simmons Street Bayonet Head", "region": "WA"},
    {"name": "DTM Tower Services", "address": "43 Martin Street Belgrave", "region": "VIC"},
    {"name": "Roadys Towing & Recovery", "address": "24 Charlesville Road Plenty", "region": "VIC"},
    {"name": "Dondio's Mechanical Repairs", "address": "35 King Street Myrtleford", "region": "VIC"},
    {"name": "AutoFix Collision - Newcastle", "address": "5 Coal Wash Drive Mayfield West", "region": "NSW"},
    {"name": "LSH Mechanical", "address": "136 Gwydir Street Moree", "region": "NSW"},
    {"name": "Monaro Automotive Services", "address": "2 Short Street Cooma", "region": "NSW"},
    {"name": "1EGAUTOLEC", "address": "89 Boree Park Road Richmond", "region": "QLD"},
    {"name": "Westside Auto Service", "address": "37 Ewing Street Bentley", "region": "WA"},
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
