import json
import logging
from typing import List, Dict
from urllib.parse import urlencode
import requests

from config import BDE_BASE_URL, BDE_PASS, BDE_TIMEOUT_SEC, BDE_USER

logger = logging.getLogger(__name__)


class BDEClient:
    """Cliente para obtener series del BDE directamente desde la API."""

    def __init__(self) -> None:
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def fetch_series(self, series_id: str) -> List[Dict]:
        """Obtiene serie directamente desde la API del BDE.

        Args:
            series_id: ID de la serie (e.g., "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M")
        """
        bde_url = self._build_bde_url(series_id)
        logger.info(f"Fetching series from BDE API: {bde_url}")
        bde_payload = self._fetch_raw_bde_payload(bde_url)
        return self._extract_obs(bde_payload)

    def preload_from_catalog(self, catalog_path: str) -> None:
        """Recorre el catálogo y consulta cada serie contra la API."""
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception as e:
            logger.error(f"Could not read catalog {catalog_path}: {e}")
            return

        for entry in catalog:
            sid = entry.get("id")
            if not sid:
                continue
            try:
                self.fetch_series(sid)
            except Exception as e:
                logger.warning(f"Failed to preload series {sid}: {e}")
    
    def _build_bde_url(self, timeseries_id: str) -> str:
        """Construye URL hacia API del BDE.
        
        Args:
            timeseries_id: ID de la serie (e.g., "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M")
        
        Returns:
            URL completa con parámetros
        """
        if not BDE_USER or not BDE_PASS:
            logger.warning("BDE credentials not configured. Set BDE_USER and BDE_PASS in .env")
        
        params = {
            "user": BDE_USER,
            "pass": BDE_PASS,
            "function": "GetSeries",
            "timeseries": timeseries_id
        }
        
        url = f"{BDE_BASE_URL}?{urlencode(params)}"
        return url

    def _fetch_raw_bde_payload(self, url: str) -> Dict:
        """Realiza request a la API del BDE y devuelve el payload crudo."""
        try:
            r = self._session.get(url, timeout=BDE_TIMEOUT_SEC)
            r.raise_for_status()
            
            if not r.text.strip():
                logger.warning("BDE API returned empty response. Check credentials (BDE_USER, BDE_PASS)")
                return {}
            
            return r.json()
        except Exception as e:
            logger.error(f"Failed to fetch series from BDE: {e}")
            raise

    def _extract_obs(self, payload) -> List[Dict]:
        """Extract observations from different payload formats.
        
        Supports:
        - Local format: {"obs": [{"date": "31-12-2025", "value": 112.5}]}
        - Simple store: [{"date": "31-12-2025", "value": 112.5}]
        - BDE format: {"Series": {"Obs": [{"indexDateString": "01-01-1996", "value": "42.48"}]}}
        """
        if isinstance(payload, list):
            # Already a list of observations
            return payload
        if isinstance(payload, dict):
            # Local format: payload["obs"]
            if "obs" in payload and isinstance(payload["obs"], list):
                return payload["obs"]
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
            
            # BDE format: payload["Series"]["Obs"]
            if "Series" in payload and isinstance(payload["Series"], dict):
                series = payload["Series"]
                if "Obs" in series and isinstance(series["Obs"], list):
                    # Normalize BDE format to local format
                    normalized = []
                    for obs in series["Obs"]:
                        if obs.get("statusCode") == "OK":
                            # Convert "01-01-1996" to "01-01-1996" (keep as is)
                            # Convert value string to float
                            normalized.append({
                                "date": obs["indexDateString"],
                                "value": float(obs["value"])
                            })
                    return normalized
        
        logger.warning("Unexpected payload shape; returning empty list.")
        return []
