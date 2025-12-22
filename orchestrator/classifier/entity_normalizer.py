"""
Normalizador de entidades extraídas por JointBERT.

Aplica normalización a las entidades detectadas (indicadores, períodos, componentes, estacionalidad)
usando los módulos de normalización disponibles.
"""

import logging
import re
from datetime import date
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Importar normalizadores opcionales
try:
    from orchestrator.utils.period_normalizer import standardize_imacec_time_ref
except ImportError:
    standardize_imacec_time_ref = None
    logger.warning("period_normalizer no disponible")

try:
    from orchestrator.utils.indicator_normalizer import standardize_indicator
except ImportError:
    standardize_indicator = None
    logger.warning("indicator_normalizer no disponible")

try:
    from orchestrator.utils.component_normalizer import normalize_component
except ImportError:
    normalize_component = None
    logger.warning("component_normalizer no disponible")

try:
    from orchestrator.utils.seasonality_normalizer import normalize_seasonality
except ImportError:
    normalize_seasonality = None
    logger.warning("seasonality_normalizer no disponible")


class EntityNormalizer:
    """Normaliza entidades extraídas de JointBERT."""
    
    def __init__(self):
        self.period_normalizer = standardize_imacec_time_ref
        self.indicator_normalizer = standardize_indicator
        self.component_normalizer = normalize_component
        self.seasonality_normalizer = normalize_seasonality
        
        # Palabras clave de estacionalidad
        self.seasonality_keywords = [
            "desestacionalizado", "desestacionalizada", "desestacionalizados", "desestacionalizadas",
            "ajustado por estacionalidad", "ajustada por estacionalidad", 
            "ajustados por estacionalidad", "ajustadas por estacionalidad",
            "no desestacionalizado", "no desestacionalizada", 
            "original", "sin ajuste estacional", "sin ajustar"
        ]
    
    def normalize(self, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Normaliza todas las entidades detectadas.
        
        Args:
            entities: Dict con entidades sin normalizar
        
        Returns:
            Dict con entidades normalizadas
        """
        normalized = {}
        
        # Normalizar período
        if 'period' in entities:
            period_result = self._normalize_period(entities['period'])
            if period_result:
                normalized['period'] = period_result
        
        # Normalizar indicador
        if 'indicator' in entities:
            indicator_result = self._normalize_indicator(entities['indicator'])
            if indicator_result:
                normalized['indicator'] = indicator_result
        
        # Separar y normalizar component/seasonality si aparecen juntos
        if 'component' in entities:
            comp_clean, seas = self._split_component_seasonality(entities['component'])
            if seas:
                entities['component'] = comp_clean
                entities['seasonality'] = seas
        
        # Normalizar sector/componente
        if 'sector' in entities or 'component' in entities:
            sector_text = entities.get('sector') or entities.get('component')
            component_result = self._normalize_component(sector_text)
            if component_result:
                normalized['component'] = component_result
                normalized['sector'] = component_result  # Mantener compatibilidad
        
        # Normalizar estacionalidad
        if 'seasonality' in entities:
            seasonality_result = self._normalize_seasonality(entities['seasonality'])
            if seasonality_result:
                normalized['seasonality'] = seasonality_result
        
        return normalized
    
    def _normalize_period(self, period: str) -> Optional[Dict[str, Any]]:
        """Normaliza período usando period_normalizer."""
        if not self.period_normalizer:
            logger.warning("period_normalizer no está disponible")
            return None
        
        try:
            result = self.period_normalizer(period, date.today())
            return result if result else None
        except Exception as e:
            logger.warning(f"Error normalizando período '{period}': {e}", exc_info=True)
            return None
    
    def _normalize_indicator(self, indicator: str) -> Optional[Dict[str, Any]]:
        """Normaliza indicador usando indicator_normalizer."""
        if not self.indicator_normalizer:
            logger.warning("indicator_normalizer no está disponible")
            return None
        
        try:
            result = self.indicator_normalizer(indicator)
            if result and result.get('indicator'):
                indicator_value = result['indicator']
                return {
                    'standard_name': indicator_value,
                    'normalized': indicator_value,
                    'text_normalized': result['text_norm'],
                    'detected_by': result.get('text_standardized_imacec')
                }
            return None
        except Exception as e:
            logger.warning(f"Error normalizando indicador '{indicator}': {e}", exc_info=True)
            return None
    
    def _normalize_component(self, component: str) -> Optional[Dict[str, Any]]:
        """Normaliza componente/sector usando component_normalizer."""
        if not self.component_normalizer:
            return None
        
        try:
            result = self.component_normalizer(component)
            logger.info(f"Componente normalizado: {result}")
            return {
                'standard_name': result,
                'normalized': result,
                'original': component
            }
        except Exception as e:
            logger.warning(f"Error normalizando componente '{component}': {e}", exc_info=True)
            return None
    
    def _normalize_seasonality(self, seasonality: str) -> Optional[str]:
        """Normaliza estacionalidad usando seasonality_normalizer."""
        if not self.seasonality_normalizer:
            return None
        
        try:
            result = self.seasonality_normalizer(seasonality)
            logger.info(f"Estacionalidad normalizada: {result}")
            return result
        except Exception as e:
            logger.warning(f"Error normalizando estacionalidad '{seasonality}': {e}", exc_info=True)
            return None
    
    def _split_component_seasonality(self, text: str) -> tuple:
        """
        Separa component y seasonality si aparecen juntos.
        
        Returns:
            (component_clean, seasonality) o (text, None) si no hay separación
        """
        for kw in self.seasonality_keywords:
            pattern = r"\b" + re.escape(kw) + r"[\b\?!.,;:]*$"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                idx = match.start()
                comp = text[:idx].strip(" ,;:¿?¡!.")
                seas = text[idx:].strip(" ,;:¿?¡!.")
                return comp, seas
        return text, None


# Instancia global del normalizer
_entity_normalizer: Optional[EntityNormalizer] = None


def get_entity_normalizer() -> EntityNormalizer:
    """
    Obtiene la instancia global del normalizer (singleton).
    
    Returns:
        Instancia de EntityNormalizer
    """
    global _entity_normalizer
    if _entity_normalizer is None:
        _entity_normalizer = EntityNormalizer()
    return _entity_normalizer


def normalize_entities(entities: Dict[str, str]) -> Dict[str, Any]:
    """
    Función de conveniencia para normalizar entidades.
    
    Args:
        entities: Dict con entidades sin normalizar
    
    Returns:
        Dict con entidades normalizadas
    
    Ejemplo:
        entities = {'indicator': 'imacec', 'period': 'agosto 2024'}
        normalized = normalize_entities(entities)
    """
    normalizer = get_entity_normalizer()
    return normalizer.normalize(entities)
