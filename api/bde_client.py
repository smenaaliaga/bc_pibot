import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Iterable
import requests

logger = logging.getLogger(__name__)


class BDEClient:
    """Cliente para obtener series del BDE con almacén único en disco.

    - Almacén persistente: `data_store/timeseries.json`, mapeando `series_id -> payload completo`
    - Carga el almacén completo en memoria al iniciar y sirve desde allí
    - Si una serie falta o está vacía, intenta local y luego BDE, y persiste en el almacén
    """

    def __init__(self):
        # Ruta del almacén único
        base_store = Path(__file__).resolve().parent.parent / "data_store"
        base_store.mkdir(parents=True, exist_ok=True)
        self._store_path = base_store / "timeseries.json"
        # Mapa en memoria: series_id -> payload (BDE-like)
        self._store: Dict[str, Dict] = {}
        if self._store_path.exists():
            try:
                with self._store_path.open("r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load store file {self._store_path}: {e}. Starting with empty store.")
                self._store = {}

    def fetch_series(self, series_id: str) -> List[Dict]:
        """Obtiene serie desde almacén único, local, o BDE y persiste al almacén.

        Args:
            series_id: ID de la serie (e.g., "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M")
        """
        # 1) Intentar desde almacén en memoria
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
        
        # 2) Intentar carga local si existe archivo con ese nombre
        fname = f"{series_id}.json"
        path = os.path.join(os.path.dirname(__file__), "..", "sample_data", fname)
        path = os.path.abspath(path)
        if os.path.exists(path):
            logger.info(f"Loading local sample series: {path}")
            print(f"   Loading from local file: {fname}")
            with open(path, "r", encoding="utf-8") as f:
                local_payload = json.load(f)
            obs = self._extract_obs(local_payload)
            # Persistimos el payload local tal cual
            self._persist_store(series_id, local_payload)
            return obs
        
        # Si no existe localmente, intentar desde BDE API
        # print(f"   Fetching from BDE API...")
        logger.info(f"Fetching series from BDE API: {bde_url}")
        try:
            bde_timeout = int(os.getenv("BDE_TIMEOUT_SEC", "15"))
            r = requests.get(bde_url, timeout=bde_timeout)
            r.raise_for_status()
            if not r.text.strip():
                logger.warning("BDE API returned empty response. Check credentials (BC_PIBOT_BDE_USER, BC_PIBOT_BDE_PASS)")
                return []
            bde_payload = r.json()
        except Exception as e:
            logger.error(f"Failed to fetch series from BDE: {e}")
            raise
        obs = self._extract_obs(bde_payload)
        # Persistimos el payload BDE tal cual
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
            print(f"   WARNING: BDE credentials not set (user='{bde_user}', pass='{bde_pass}')")
        
        params = {
            "user": bde_user,
            "pass": bde_pass,
            "function": "GetSeries",
            "timeseries": timeseries_id
        }
        
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{bde_base_url}?{query_string}"
        # print(f"BDE URL: {url}")
        return url

    def _persist_store(self, series_id: str, payload: Dict) -> None:
        # Persiste en memoria y en disco el payload completo (tal cual fuente)
        self._store[series_id] = payload
        try:
            with self._store_path.open("w", encoding="utf-8") as f:
                # Compacto: sin espacios innecesarios
                json.dump(self._store, f, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            logger.warning(f"Could not persist store to {self._store_path}: {e}")
    
    def _fetch_from_bde(self, url: str) -> List[Dict]:
        """Realiza request a la API del BDE."""
        try:
            bde_timeout = int(os.getenv("BDE_TIMEOUT_SEC", "15"))
            r = requests.get(url, timeout=bde_timeout)
            r.raise_for_status()
            
            if not r.text.strip():
                logger.warning("BDE API returned empty response. Check credentials (BC_PIBOT_BDE_USER, BC_PIBOT_BDE_PASS)")
                return []
            
            payload = r.json()
            return self._extract_obs(payload)
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
                    logger.info(f"Extracted {len(normalized)} observations from BDE format")
                    return normalized
        
        logger.warning("Unexpected payload shape; returning empty list.")
        return []
