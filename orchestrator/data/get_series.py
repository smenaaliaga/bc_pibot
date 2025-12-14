"""
Módulo para detectar series correctas basado en entidades normalizadas.

Este módulo proporciona una función modular para mapear entidades detectadas
(indicator, component, etc.) a los códigos de series del Banco Central de Chile.

Utiliza el catálogo de series (catalog/series_catalog.json) como fuente de verdad.
Los valores normalizados deben coincidir con los standard_names del catálogo.
"""

import json
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Valores por defecto
DEFAULT_INDICATOR = "imacec"
DEFAULT_COMPONENT = "imacec"

# Cache del catálogo de series
_SERIES_CATALOG: Optional[Dict[str, Any]] = None
_SERIES_INDEX: Optional[Dict[str, List[str]]] = None


def _normalize_text(text: str) -> str:
    """Normaliza texto para comparación (lowercase, sin acentos)."""
    if not text:
        return ""
    
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"
    }
    
    text = text.lower().strip()
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def _load_series_catalog() -> Dict[str, Any]:
    """Carga el catálogo de series desde el archivo JSON."""
    global _SERIES_CATALOG
    
    if _SERIES_CATALOG is not None:
        return _SERIES_CATALOG
    
    # Buscar el archivo del catálogo
    catalog_path = Path(__file__).parent.parent.parent / "catalog" / "series_catalog.json"
    
    if not catalog_path.exists():
        logger.warning(f"Catálogo no encontrado en: {catalog_path}")
        _SERIES_CATALOG = {}
        return _SERIES_CATALOG
    
    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            _SERIES_CATALOG = json.load(f)
        logger.info(f"Catálogo cargado: {len(_SERIES_CATALOG)} series")
        return _SERIES_CATALOG
    except Exception as e:
        logger.error(f"Error cargando catálogo: {e}")
        _SERIES_CATALOG = {}
        return _SERIES_CATALOG


def _build_series_index() -> Dict[str, List[str]]:
    """
    Construye un índice invertido para búsqueda rápida.
    Mapea (indicator:component) -> [series_codes]
    
    Usa los standard_names del catálogo como referencia.
    """
    global _SERIES_INDEX
    
    if _SERIES_INDEX is not None:
        return _SERIES_INDEX
    
    catalog = _load_series_catalog()
    _SERIES_INDEX = {}
    
    for series_code, metadata in catalog.items():
        # Usar standard_names como fuente de verdad
        standard_names = metadata.get("standard_names", {})
        
        indicator = standard_names.get("indicator", "")
        component = standard_names.get("component", "")
        
        if not indicator:
            continue
        
        # Crear claves de búsqueda (los standard_names ya están normalizados)
        # 1. Con componente específico
        if component:
            key = f"{indicator}:{component}"
            if key not in _SERIES_INDEX:
                _SERIES_INDEX[key] = []
            _SERIES_INDEX[key].append(series_code)
        
        # 2. Solo indicador (para búsquedas genéricas)
        key_generic = f"{indicator}:"
        if key_generic not in _SERIES_INDEX:
            _SERIES_INDEX[key_generic] = []
        _SERIES_INDEX[key_generic].append(series_code)
    
    logger.info(f"Índice construido: {len(_SERIES_INDEX)} combinaciones")
    return _SERIES_INDEX


def _select_best_series(candidates: List[str], catalog: Dict[str, Any]) -> Optional[str]:
    """
    Selecciona la mejor serie de una lista de candidatos.
    Prioriza series canónicas y con mayor prioridad.
    """
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Ordenar por prioridad y canonical
    scored = []
    for code in candidates:
        metadata = catalog.get(code, {})
        priority = metadata.get("priority", 0)
        is_canonical = metadata.get("canonical", False)
        score = priority + (1000 if is_canonical else 0)
        scored.append((score, code))
    
    scored.sort(reverse=True)
    return scored[0][1]


def _find_best_match(indicator: str, component: str) -> Optional[str]:
    """
    Busca la mejor coincidencia en el catálogo de series usando standard_names.
    
    Args:
        indicator: Indicador normalizado (ej: "imacec", "pib")
        component: Componente normalizado (ej: "minero", "no minero", "produccion de bienes")
    
    Returns:
        Código de serie o None si no se encuentra
    """
    index = _build_series_index()
    catalog = _load_series_catalog()
    
    # Los inputs ya deben venir normalizados para coincidir con standard_names
    indicator_norm = _normalize_text(indicator if indicator else "")
    component_norm = _normalize_text(component if component else "")
    
    # Estrategia 1: Búsqueda exacta con componente
    if component_norm:
        key = f"{indicator_norm}:{component_norm}"
        if key in index:
            candidates = index[key]
            return _select_best_series(candidates, catalog)
    
    # Estrategia 2: Buscar por similitud en standard_names.component
    key_indicator = f"{indicator_norm}:"
    if key_indicator in index and component_norm:
        candidates = index[key_indicator]
        for series_code in candidates:
            metadata = catalog[series_code]
            standard_names = metadata.get("standard_names", {})
            
            # Buscar en standard_names.component
            std_component = _normalize_text(standard_names.get("component", ""))
            if component_norm in std_component or std_component in component_norm:
                return series_code
            
            # Buscar en classification
            classification = _normalize_text(metadata.get("classification", ""))
            if component_norm in classification or classification in component_norm:
                return series_code
            
            # Buscar en aliases
            aliases = metadata.get("aliases", [])
            for alias in aliases:
                alias_norm = _normalize_text(alias)
                if component_norm in alias_norm:
                    return series_code
    
    # Estrategia 3: Fallback - devolver serie por defecto del indicador (mayor prioridad/canónica)
    if key_indicator in index:
        candidates = index[key_indicator]
        return _select_best_series(candidates, catalog)
    
    return None


def detect_series_code(
    normalized: Optional[Dict[str, Any]] = None,
    entities: Optional[Dict[str, Any]] = None,
    indicator: Optional[str] = None,
    component: Optional[str] = None,
    sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detecta el código de serie correcto basado en las entidades normalizadas.
    
    Esta función mapea entidades normalizadas a códigos de serie del BCCh
    usando los facets del catálogo como referencia.
    
    Args:
        normalized: Diccionario con datos normalizados del clasificador
        entities: Diccionario con entidades extraídas (alias de normalized)
        indicator: Indicador específico (sobrescribe detección automática)
        component: Componente/sector específico (sobrescribe detección) - DEPRECATED: usar sector
        sector: Sector específico (sobrescribe detección) - alineado con facets del catálogo
        
    Returns:
        Dict con:
            - series_code: Código de la serie detectada
            - indicator: Indicador detectado/usado
            - sector: Sector detectado/usado
            - metadata: Metadata de la serie del catálogo
            - matched_by: Método de detección usado
    
    Ejemplos:
        >>> detect_series_code()
        {'series_code': 'F032.IMC.IND.Z.Z.EP18.Z.Z.0.M', ...}
        
        >>> detect_series_code(indicator="imacec", sector="minero")
        {'series_code': 'F032.IMC.IND.Z.Z.EP18.03.Z.0.M', ...}
    """
    # Extraer valores de los parámetros (con prioridad a argumentos explícitos)
    if normalized is None:
        normalized = entities or {}
    
    # Extraer indicator (normalizar a lowercase)
    if indicator:
        final_indicator = _normalize_text(indicator)
    else:
        indicator_value = normalized.get("indicator")
        if isinstance(indicator_value, dict):
            # Formato: {"standard_name": "IMACEC", ...}
            final_indicator = _normalize_text(indicator_value.get("standard_name") or indicator_value.get("original") or DEFAULT_INDICATOR)
        elif isinstance(indicator_value, str):
            final_indicator = _normalize_text(indicator_value)
        else:
            final_indicator = _normalize_text(DEFAULT_INDICATOR)
    
    # Extraer sector (normalizar a lowercase)
    # Prioridad: sector > component > normalized.sector > normalized.component > default
    final_sector = None
    
    if sector:
        final_sector = _normalize_text(sector)
    elif component:
        final_sector = _normalize_text(component)
    else:
        # Intentar extraer de normalized
        sector_value = normalized.get("sector")
        if isinstance(sector_value, dict):
            final_sector = _normalize_text(sector_value.get("standard_name") or sector_value.get("original") or "")
        elif isinstance(sector_value, str):
            final_sector = _normalize_text(sector_value)
        
        # Fallback a 'component' si no hay 'sector'
        if not final_sector:
            component_value = normalized.get("component")
            if isinstance(component_value, dict):
                final_sector = _normalize_text(component_value.get("standard_name") or component_value.get("original") or "")
            elif isinstance(component_value, str):
                final_sector = _normalize_text(component_value)
        
        # Si aún no hay sector, usar default
        if not final_sector:
            final_sector = _normalize_text(DEFAULT_COMPONENT)
    
    # Buscar serie en el catálogo usando component (no sector)
    series_code = _find_best_match(final_indicator, final_sector)
    
    # Obtener metadata de la serie
    catalog = _load_series_catalog()
    metadata = catalog.get(series_code, {}) if series_code else {}
    
    # Si no se encuentra, intentar con valores por defecto
    if not series_code:
        logger.warning(
            f"No se encontró serie para indicator='{final_indicator}', component='{final_sector}'. "
            f"Usando valores por defecto."
        )
        series_code = _find_best_match(_normalize_text(DEFAULT_INDICATOR), _normalize_text(DEFAULT_COMPONENT))
        metadata = catalog.get(series_code, {}) if series_code else {}
    
    result = {
        "series_code": series_code,
        "indicator": final_indicator,
        "component": final_sector,  # Usar component como campo principal
        "sector": final_sector,      # Mantener por compatibilidad
        "metadata": metadata if metadata else {},
        "matched_by": "catalog_standard_names",
    }
    
    logger.info(
        f"Serie detectada: {series_code} | "
        f"indicator={final_indicator}, component={final_sector}"
    )
    
    return result


def add_series_mapping(
    indicator: str,
    component: str,
    series_code: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Agrega un mapeo personalizado de serie (para extensión dinámica).
    
    NOTA: Este método agrega al cache en memoria, no modifica el catálogo en disco.
    
    Args:
        indicator: Indicador (ej: "imacec")
        component: Componente/sector (ej: "servicios")
        series_code: Código de serie BCCh
        metadata: Metadata opcional de la serie
    """
    catalog = _load_series_catalog()
    
    # Agregar al catálogo en memoria
    if series_code not in catalog:
        catalog[series_code] = metadata or {}
    
    # Actualizar standard_names para que el índice lo encuentre
    if "standard_names" not in catalog[series_code]:
        catalog[series_code]["standard_names"] = {}
    
    catalog[series_code]["standard_names"]["indicator"] = indicator
    catalog[series_code]["standard_names"]["component"] = component
    
    # Invalidar índice para que se reconstruya
    global _SERIES_INDEX
    _SERIES_INDEX = None
    
    logger.info(f"Mapeo agregado: {indicator}:{component} -> {series_code}")


def get_available_mappings() -> Dict[str, List[str]]:
    """
    Obtiene todos los mapeos disponibles en el catálogo.
    
    Returns:
        Diccionario {indicator: [components]}
    """
    index = _build_series_index()
    result = {}
    
    for key in index.keys():
        if ":" in key:
            indicator, component = key.split(":", 1)
            if indicator not in result:
                result[indicator] = []
            if component and component not in result[indicator]:
                result[indicator].append(component)
    
    return result


def get_series_info(series_code: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene información completa de una serie del catálogo.
    
    Args:
        series_code: Código de serie BCCh
    
    Returns:
        Metadata de la serie o None si no existe
    """
    catalog = _load_series_catalog()
    return catalog.get(series_code)
