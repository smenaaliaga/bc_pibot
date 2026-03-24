import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlencode
import requests

logger = logging.getLogger(__name__)


class BDEClient:
    """Cliente para obtener series del BDE con almacén único en disco.

    - Almacén persistente: `data_store/timeseries.json`, mapeando `series_id -> payload completo`
    - Carga el almacén completo en memoria al iniciar y sirve desde allí
    - Si una serie falta o está vacía, consulta BDE y persiste en el almacén
    """

    def __init__(self, store_dir: Path = None, force_api: bool = False):
        if store_dir is not None:
            base_store = Path(store_dir)
        else:
            base_store = Path(__file__).resolve().parents[2] / "data_store"
        base_store.mkdir(parents=True, exist_ok=True)
        self._store_path = base_store / "timeseries.json"
        # Si está activo, siempre consulta API y omite lectura desde caché.
        self._force_api = force_api
        # Mapa en memoria: series_id -> payload (BDE-like)
        self._store: Dict[str, Any] = {}
        if self._store_path.exists():
            try:
                with self._store_path.open("r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load store file {self._store_path}: {e}. Starting with empty store.")
                self._store = {}

    def fetch_series(self, series_id: str) -> List[Dict]:
        """Obtiene serie desde almacén único o BDE y persiste al almacén.

        Args:
            series_id: ID de la serie (e.g., "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M")
        """
        # 1) Intentar desde almacén en memoria (salvo force_api)
        if not self._force_api:
            payload = self._store.get(series_id)
            if isinstance(payload, list):
                if payload:
                    logger.info(f"Serving series from in-memory store: {series_id}")
                    return payload
            elif isinstance(payload, dict):
                obs = self._extract_obs(payload)
                if obs:
                    logger.info(f"Serving series from in-memory store: {series_id}")
                    return obs

        # Mostrar URL que se usaría para BDE
        bde_url = self._build_bde_url(series_id)
        
        # 2) Obtener desde BDE API
        logger.info(f"Fetching series from BDE API: {bde_url}")
        bde_payload = self._fetch_raw_bde_payload(bde_url)
        obs = self._extract_obs(bde_payload)
        # En modo force_api evitamos cachear en memoria/disco.
        if not self._force_api:
            self._persist_store(series_id, bde_payload)
        return obs

    def preload_from_catalog(self, catalog_path: str) -> None:
        """Recorre el catálogo y llena el almacén único en disco/memoria."""
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
                # Evita llamadas si ya está en almacén con datos
                payload = self._store.get(sid)
                obs = self._extract_obs(payload) if payload else []
                if obs:
                    continue
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
        bde_user = os.getenv("BDE_USER", "")
        bde_pass = os.getenv("BDE_PASS", "")
        bde_base_url = os.getenv("BDE_BASE_URL", "https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx")
        
        if not bde_user or not bde_pass:
            logger.warning("BDE credentials not configured. Set BDE_USER and BDE_PASS in .env")
        
        params = {
            "user": bde_user,
            "pass": bde_pass,
            "function": "GetSeries",
            "timeseries": timeseries_id
        }
        
        url = f"{bde_base_url}?{urlencode(params)}"
        return url

    def _persist_store(self, series_id: str, payload: Any) -> None:
        # Persiste en memoria y en disco el payload completo (tal cual fuente)
        self._store[series_id] = payload
        try:
            with self._store_path.open("w", encoding="utf-8") as f:
                # Compacto: sin espacios innecesarios
                json.dump(self._store, f, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            logger.warning(f"Could not persist store to {self._store_path}: {e}")
    
    def _fetch_raw_bde_payload(self, url: str) -> Dict:
        """Realiza request a la API del BDE y devuelve el payload crudo."""
        try:
            bde_timeout = int(os.getenv("BDE_TIMEOUT_SEC", "15"))
            r = requests.get(url, timeout=bde_timeout)
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
