import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CatalogService:
    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        with open(catalog_path, "r", encoding="utf-8") as f:
            self._entries = json.load(f)
        # logger.info(f"Loaded catalog with {len(self._entries)} entries from {catalog_path}")

    def find_series(self, indicator: str, metric_type: str, seasonality: str, activity: str, frequency: str) -> Optional[Dict]:
        for entry in self._entries:
            # Skip entries without 'id' (e.g., header rows)
            if "id" not in entry:
                continue
            
            # Extract classification from nested structure
            classification = entry.get("classification", {})
            
            # Match all classification dimensions
            if (
                classification.get("indicator") == indicator
                and classification.get("metric_type") == metric_type
                and classification.get("seasonality") == seasonality
                and classification.get("activity") == activity
                and classification.get("frequency") == frequency
            ):
                logger.info(f"Matched catalog series: {entry.get('id')}")
                return entry
        
        logger.warning(f"No series matched for: indicator={indicator}, metric_type={metric_type}, seasonality={seasonality}, activity={activity}, frequency={frequency}")
        return None
