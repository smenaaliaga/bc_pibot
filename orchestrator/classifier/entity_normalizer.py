"""
Normalizador de entidades extraídas por JointBERT.

Aplica normalización a las entidades detectadas (indicadores, períodos, componentes, estacionalidad)
usando los módulos de normalización disponibles.
"""

import logging
import re
import calendar
from datetime import date, datetime
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
        Normaliza todas las entidades detectadas, devolviendo una estructura simplificada.
        
        Args:
            entities: Dict con entidades sin normalizar (raw output de JointBERT)
        
        Returns:
            Dict con entidad normalizada lista para consumo por `flow_data`:
            
            {
                'period': {
                    'start_date': date,           # inicio del período
                    'end_date': date,             # fin del período (fin de mes/trimestre/año)
                    'granularity': str,           # 'month' | 'quarter' | 'year'
                    'period_type': str,           # alias de granularity cuando el normalizador lo entregue
                    'period_key': str,            # '2024-08' o '2024-Q3'
                    'label': str,                 # 'Agosto 2024', '3T 2024'
                },
                'indicator': {
                    'normalized': str,
                },
                'component': {
                    'normalized': str,
                },
                'sector': {
                    'normalized': str,
                },
                'seasonality': str                # 'desestacionalizado' o 'original'
            }
            
            Notas:
            - Solo se incluyen claves para entidades detectadas
            - 'component' y 'sector' son sinónimos y comparten el mismo valor normalizado
        """
        normalized = {}
        
        # Normalizar período (estructura completa útil para obtención de serie)
        if 'period' in entities:
            period_result = self._normalize_period(entities['period'])
            if period_result:
                # Enriquecer con ventana de consulta y candidatos
                normalized['period'] = self._enrich_period_window(period_result)
        
        # Normalizar indicador (solo 'normalized')
        if 'indicator' in entities:
            indicator_result = self._normalize_indicator(entities['indicator'])
            if indicator_result:
                normalized['indicator'] = {'normalized': indicator_result}
        
        # Separar y normalizar component/seasonality si aparecen juntos
        if 'component' in entities:
            comp_clean, seas = self._split_component_seasonality(entities['component'])
            if seas:
                entities['component'] = comp_clean
                entities['seasonality'] = seas
        
        # Normalizar sector/componente (solo 'normalized')
        if 'sector' in entities or 'component' in entities:
            sector_text = entities.get('sector') or entities.get('component')
            component_result = self._normalize_component(sector_text)
            if component_result:
                comp_val = {'normalized': component_result}
                normalized['component'] = comp_val
                normalized['sector'] = comp_val
        
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

    def _coerce_date(self, value: Any) -> Optional[date]:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if value in (None, ""):
            return None
        text = str(value).strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
            try:
                dt = datetime.strptime(text, fmt)
                if fmt == "%Y-%m":
                    # asumir fin de mes como end_date
                    last_day = calendar.monthrange(dt.year, dt.month)[1]
                    return date(dt.year, dt.month, 1)
                if fmt == "%Y":
                    return date(dt.year, 1, 1)
                return dt.date()
            except Exception:
                continue
        return None

    def _end_of_period(self, value: date, granularity: str) -> date:
        gran = (granularity or "").lower()
        if gran == "month":
            last_day = calendar.monthrange(value.year, value.month)[1]
            return value.replace(day=last_day)
        if gran == "quarter":
            quarter = ((value.month - 1) // 3) + 1
            last_month = quarter * 3
            last_day = calendar.monthrange(value.year, last_month)[1]
            return value.replace(month=last_month, day=last_day)
        if gran == "year":
            return value.replace(month=12, day=31)
        return value

    def _subtract_months(self, value: date, months: int) -> date:
        total_months = value.year * 12 + (value.month - 1) - max(months, 0)
        year = total_months // 12
        month = total_months % 12 + 1
        last_day = calendar.monthrange(year, month)[1]
        day = min(value.day, last_day)
        return value.replace(year=year, month=month, day=day)

    def _format_date(self, value: Optional[date]) -> Optional[str]:
        return value.strftime("%Y-%m-%d") if value else None

    def _date_from_period_key(self, period_key: Any) -> Optional[str]:
        if not period_key:
            return None
        key = str(period_key).strip()
        if re.match(r"^\d{4}-\d{2}$", key):
            try:
                y = int(key[:4])
                m = int(key[-2:])
                last_day = calendar.monthrange(y, m)[1]
                return f"{y:04d}-{m:02d}-{last_day:02d}"
            except Exception:
                return None
        if re.match(r"^\d{4}-Q[1-4]$", key.upper()):
            year = int(key[:4])
            quarter = int(key[-1])
            month = quarter * 3
            last_day = calendar.monthrange(year, month)[1]
            return f"{year:04d}-{month:02d}-{last_day:02d}"
        return None

    def _enrich_period_window(self, period_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agrega 'firstdate', 'lastdate' y 'match_candidates' al período normalizado para
        consumo directo por el flujo de datos.
        """
        raw_start = period_result.get("start_date") or period_result.get("startDate")
        raw_end = period_result.get("end_date") or period_result.get("endDate")
        gran = str(period_result.get("granularity") or period_result.get("period_type") or "").lower()

        start = self._coerce_date(raw_start)
        end = self._coerce_date(raw_end)
        if not start and end:
            start = end
        if start and not end:
            end = self._end_of_period(start, gran)

        # Si no hay fechas, intentar derivar desde period_key
        if not start and not end:
            key_date = self._date_from_period_key(period_result.get("period_key"))
            if key_date:
                try:
                    end_dt = datetime.strptime(key_date, "%Y-%m-%d").date()
                    end = end_dt
                    # Inferir granularidad por el formato del key
                    pk = str(period_result.get("period_key") or "")
                    if re.match(r"^\d{4}-\d{2}$", pk):
                        gran = "month"
                        start = date(end_dt.year, end_dt.month, 1)
                    elif re.match(r"^\d{4}-Q[1-4]$", pk.upper()):
                        gran = "quarter"
                        # calcular inicio de trimestre
                        q = int(pk[-1])
                        first_month = (q - 1) * 3 + 1
                        start = date(end_dt.year, first_month, 1)
                    else:
                        gran = gran or "year"
                        start = date(end_dt.year, 1, 1)
                except Exception:
                    pass

        # Construir ventana de consulta
        target = end or start
        firstdate = None
        lastdate = None
        candidates: list[str] = []
        if target:
            buffer_months = {"month": 15, "quarter": 24, "year": 60}.get(gran, 18)
            first_candidate = self._subtract_months(target, buffer_months)
            firstdate = self._format_date(first_candidate)
            lastdate = self._format_date(target)
        for dt in (start, end):
            formatted = self._format_date(dt)
            if formatted:
                candidates.append(formatted)

        key_date = self._date_from_period_key(period_result.get("period_key"))
        if key_date:
            candidates.append(key_date)

        label = period_result.get("label") or period_result.get("period_key") or lastdate or firstdate

        # Escribir de vuelta
        if firstdate:
            period_result["firstdate"] = firstdate
        if lastdate:
            period_result["lastdate"] = lastdate
        period_result["granularity"] = gran or period_result.get("granularity")
        period_result["match_candidates"] = [c for c in candidates if c]
        period_result["label"] = label

        return period_result
    
    def _normalize_indicator(self, indicator: str) -> Optional[str]:
        """Normaliza indicador y devuelve el valor normalizado simple."""
        if not self.indicator_normalizer:
            logger.warning("indicator_normalizer no está disponible")
            return None
        
        try:
            result = self.indicator_normalizer(indicator)
            if result and result.get('indicator'):
                return result['indicator']
            return None
        except Exception as e:
            logger.warning(f"Error normalizando indicador '{indicator}': {e}", exc_info=True)
            return None
    
    def _normalize_component(self, component: str) -> Optional[str]:
        """Normaliza componente/sector y devuelve valor normalizado simple."""
        if not self.component_normalizer:
            return None
        
        try:
            result = self.component_normalizer(component)
            return result
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
        entities: Dict con entidades sin normalizar (output de JointBERT)
                  Ejemplo: {'indicator': 'imacec', 'period': 'agosto 2024'}
    
    Returns:
        Dict con entidades normalizadas según estructura documentada en EntityNormalizer.normalize()
        
        Campos comunes en flow_data.py:
        - normalized['indicator']: dict con 'standard_name', 'normalized', 'text_normalized'
        - normalized['period']: dict con 'start_date', 'end_date', 'granularity', 'period_key'
        - normalized['seasonality']: str ('desestacionalizado' o 'original')
        - normalized['component']/['sector']: dict con 'standard_name', 'normalized', 'original'
    
    Ejemplo:
        >>> entities = {'indicator': 'imacec', 'period': 'agosto 2024'}
        >>> normalized = normalize_entities(entities)
        >>> normalized['indicator']['standard_name']  # 'imacec'
        >>> normalized['period']['period_key']         # '2024-08'
    """
    normalizer = get_entity_normalizer()
    return normalizer.normalize(entities)
