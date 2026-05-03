#!/usr/bin/env python3
"""Add source_url from series_catalog_BKP.json to series_catalog.json"""

import json
import sys
from pathlib import Path

def main():
    # Paths
    base_path = Path(__file__).parent.parent
    catalog_path = base_path / "catalog" / "series_catalog.json"
    bkp_path = base_path / "catalog" / "old" / "series_catalog_BKP.json"
    
    # Load both files
    print(f"Loading {catalog_path}...")
    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    
    print(f"Loading {bkp_path}...")
    with open(bkp_path, 'r', encoding='utf-8') as f:
        bkp = json.load(f)
    
    # Build lookup dict from BKP (ID -> source_url)
    source_urls = {}
    for series_id, series_data in bkp.items():
        if isinstance(series_data, dict) and "source_url" in series_data:
            source_urls[series_id] = series_data["source_url"]
    
    print(f"Found {len(source_urls)} source URLs in BKP file")
    
    # Update catalog entries
    updated_count = 0
    for entry in catalog:
        series_id = entry.get("id")
        if series_id and series_id in source_urls:
            entry["source_url"] = source_urls[series_id]
            updated_count += 1
    
    print(f"Updated {updated_count} entries with source_url")
    
    # Write back
    print(f"Writing updated catalog to {catalog_path}...")
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
