"""
============================================================================
POST-CLASSIFIER · Reglas de negocio por tipo de consulta
============================================================================

Cada regla está agrupada en una clase ``RuleNN_<Nombre>`` que reúne:

    LOGICA GENERAL  → entry-point que orquesta los métodos.
    Methods         → métodos privados con la lógica fina.
    Functions       → utilidades específicas de la regla (si aplica).
    Exceptions      → casos especiales que se ignoran o redirigen.
    Regex           → patrones compilados que pertenecen a la regla.

ORDEN DE LECTURA (también orden de invocación en ``apply_business_rules``):

    Rule10_Participaciones           (share)
    Rule02_VariacionesDesest         (prev_period / seasonality=sa)
    Rule01_ContribucionGrupal        (force general / demanda interna)
    Rule03_ValidacionPeriodoIMACEC   (frecuencia mensual)
    Rule07_ContribucionIndividual    (default activity=imacec)
    Rule09_NivelesNominales          (price_ent / price)
    Rule13_Historicos                (pisos PIB 1960 / IMACEC 1996)
    Rule04_CrecimientoPIB            (PIB mensual → trimestral)

REGLAS DE RUTEO (no mutan series, las usa ingest.py):

    Rule05_FueraDeAlcance            (out-of-scope)
    Rule06_Calendario                (calendar override)
    RuleGreeting                     (saludos)

NOTA: la categoría CAT00 *metodológica* NO está aquí (no muta entidades de
serie; se enruta por RAG en otro nodo).

Aliases planos (regex y funciones ``_rule_*``) se exponen al final del
archivo para preservar compatibilidad con:
    - ``orchestrator.data._business_rules`` (shim de re-export)
    - ``orchestrator.graph.nodes.ingest`` (predicates de ruteo)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# INFRAESTRUCTURA COMÚN
# ============================================================================

def _ensure_text(s: Any) -> str:
    return str(s or "")


def _is_empty_value(value: Any) -> bool:
    if value in (None, "", "none", "None", "null", "NULL"):
        return True
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return True
    return False


def _is_empty_cls(value: Any) -> bool:
    return value in (None, "none", "general", "", {}, [], ())


def _extract_year(token: Any) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(token or ""))
    return int(m.group(0)) if m else None


@dataclass
class ResolvedEntities:
    """Contenedor mutable con las entidades ajustadas por reglas de negocio."""

    indicator_ent: Optional[str] = None
    seasonality_ent: Optional[str] = None
    frequency_ent: Optional[str] = None
    activity_ent: Optional[str] = None
    region_ent: Optional[str] = None
    investment_ent: Optional[str] = None
    price_ent: Optional[str] = None
    period_ent: List[Any] = field(default_factory=list)

    calc_mode_cls: Any = None
    activity_cls: Any = None
    activity_cls_resolved: Any = None
    region_cls: Any = None
    investment_cls: Any = None
    req_form_cls: Any = None

    # ------------------------------------------------------------------
    # intent_cls
    # ------------------------------------------------------------------
    # Etiqueta del head ``intent`` del clasificador (uvicorn /predict →
    # ``ClassificationResult.intent``). Valores típicos:
    #   - "value"      → consulta por el NIVEL/MONTO de la serie
    #                    (ej. "cuál es el valor del IMACEC", "monto del PIB")
    #   - "variation"  → consulta por una variación (yoy/pct/aceleración)
    #   - "share"      → consulta por participación (% sobre el PIB)
    #   - "contribution" / "rank" / "extrema" / "metadata" → otros tipos
    #
    # Se propaga desde ``data_node`` (orchestrator/graph/nodes/data.py)
    # leyendo ``classification.intent`` y se consume en
    # ``orchestrator/data/response.py::_is_level_only_query`` para decidir
    # de forma SEMÁNTICA (no léxica) si la pregunta es de nivel.
    # Antes de este campo la decisión dependía de regex sobre el texto crudo,
    # lo que generaba falsos negativos con paráfrasis ("cifra", "monto",
    # "cuánto fue", "dame el ..."). El head ``intent`` cubre esas paráfrasis
    # con confianza > 0.99 según los reportes del clasificador.
    #
    # Default ``None`` para compatibilidad: si el campo no se setea (p. ej.
    # tests que construyen ResolvedEntities a mano), el comportamiento
    # vuelve al fallback léxico previo.
    intent_cls: Any = None

    price: Optional[str] = None
    hist: Optional[int] = None
    historical_floor_instruction: Optional[str] = None
    question: Optional[str] = None

    # BUG-A: dirección temporal ("earliest" = primer dato histórico disponible)
    date_direction: Optional[str] = None        # "earliest" | None

    # BUG-B2: nota de alias de región ambiguo (se propaga a response.py)
    region_alias_note: Optional[str] = None     # "antartica_standalone" | None

    applied_rules: List[Dict[str, Any]] = field(default_factory=list)


def _trace(ent: ResolvedEntities, cat: str, name: str, msg: str) -> None:
    """Registro estandarizado: [RULE] CAT_ID nombre | trace."""
    logger.info("[RULE] %s %s | %s", cat, name, msg)
    ent.applied_rules.append({"category": cat, "name": name, "trace": msg})


# Hints de actividad → indicador (usados por ingest.py para deducción)
_PIB_ACTIVITY_HINTS = {
    "agropecuario", "pesca", "industria", "electricidad", "construccion",
    "construcción", "restaurantes", "transporte", "comunicaciones",
    "servicio_financieros", "servicios_financieros", "servicios_empresariales",
    "servicio_viviendas", "servicios_vivienda", "servicio_personales",
    "servicios_personales", "admin_publica", "administracion_publica",
    "administración_pública", "impuestos",
}
_IMACEC_ACTIVITY_HINTS = {
    "bienes", "mineria", "minería", "industria", "resto_bienes",
    "servicios", "no_mineria", "no_minería",
}
_PREVIOUS_ACTIVITY_HINTS = {"comercio", "impuesto"}


# Hints léxicos compartidos: el usuario pide explícitamente el NIVEL/MONTO
# de una serie (no su variación). Usado por Rule14 (PIB regional) y Rule18
# (PIB por actividad) para NO forzar a yoy cuando hay intención de nivel.
_LEVEL_HINTS_RE = re.compile(
    r"\b(monto|nivel(?:es)?|valor(?:es)?\s+en\s+pesos?|"
    r"miles\s+de\s+millones|cu[aá]nt[oa]s?\s+pesos?|en\s+pesos?\b|"
    r"a\s+cu[aá]nto\s+ascien)",
    re.IGNORECASE,
)


# ============================================================================
# TIPO DE CONSULTA: 01 — CONTRIBUCIÓN GRUPAL
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Cuando la consulta pide la contribución de grupos (ej. "qué actividades
#   contribuyeron al PIB"), pero el clasificador detectó una inversión
#   *específica* sin región ni inversión normalizada, se fuerza el desglose
#   completo (activity_cls_resolved='general'). Adicionalmente, si la
#   inversión es 'demanda_interna', se reclasifica a 'general' para abrir el
#   ranking por componentes.
#
# Input   : ent.calc_mode_cls=='contribution', ent.investment_cls=='specific'
# Output  : ent.activity_cls_resolved='general'  /  ent.investment_cls='general'
# ============================================================================


class Rule01_ContribucionGrupal:
    """REGLA_01_CONTRIBUCION_GRUPAL."""

    CAT = "CAT01"

    # ------------------------------------------------------------------
    # LOGICA GENERAL
    # ------------------------------------------------------------------
    @classmethod
    def force_general(cls, ent: ResolvedEntities) -> None:
        if not cls._matches_force_general(ent):
            return
        ent.activity_cls_resolved = "general"
        _trace(ent, cls.CAT, "contribution_force_general",
               "activity_cls_resolved=general")

    @classmethod
    def demanda_interna(cls, ent: ResolvedEntities) -> None:
        if not cls._matches_demanda_interna(ent):
            return
        ent.investment_cls = "general"
        _trace(ent, cls.CAT, "contribution_demanda_interna",
               "investment_cls=general")

    # ------------------------------------------------------------------
    # Methods (predicates)
    # ------------------------------------------------------------------
    @staticmethod
    def _matches_force_general(ent: ResolvedEntities) -> bool:
        return (
            ent.calc_mode_cls == "contribution"
            and ent.investment_cls == "specific"
            and ent.investment_cls in (None, "none")
            and ent.region_cls in (None, "none")
        )

    @staticmethod
    def _matches_demanda_interna(ent: ResolvedEntities) -> bool:
        return (
            ent.calc_mode_cls == "contribution"
            and ent.investment_cls == "specific"
            and ent.investment_ent == "demanda_interna"
        )

    # Functions    : (none)
    # Exceptions   : (none)
    # Regex        : (none)


# ============================================================================
# TIPO DE CONSULTA: 02 — VARIACIONES / DESESTACIONALIZADO
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Variaciones en frecuencia natural ("variación mensual" para IMACEC,
#   "variación trimestral" para PIB) refieren al período anterior (t-1), no
#   a la interanual. Cuando la pregunta pide eso explícitamente, o cuando
#   la serie es desestacionalizada y el calc_mode es ambiguo, forzamos
#   calc_mode='prev_period'.
#
#   También cubre el override de RUTEO usado por ingest.py: detección de
#   intención "valor desestacionalizado" → routing data (no methodology).
#
# Input   : ent.question, ent.indicator_ent, ent.seasonality_ent
# Output  : ent.calc_mode_cls='prev_period', ent.seasonality_ent='sa'
#           (y eventualmente ent.indicator_ent inferido del verbo)
# ============================================================================


class Rule02_VariacionesDesest:
    """REGLA_02_VARIACIONES_DESEST."""

    CAT = "CAT02"

    # ---- Regex --------------------------------------------------------
    YOY_KEYWORDS_RE = re.compile(
        r"\b("
        r"interanual(?:es)?|"
        r"a[nñ]o\s+anterior|"
        r"mismo\s+per[ií]odo\s+del\s+a[nñ]o|"
        r"mismo\s+mes\s+del\s+a[nñ]o|"
        r"mismo\s+trimestre\s+del\s+a[nñ]o|"
        r"12\s*meses|doce\s+meses|"
        r"respecto\s+al\s+a[nñ]o\s+anterior|"
        r"variaci[oó]n\s+anual"
        r")\b",
        re.IGNORECASE,
    )
    VAR_MENSUAL_RE = re.compile(r"variaci[oó]n\s+mensual", re.IGNORECASE)
    VAR_TRIMESTRAL_RE = re.compile(r"variaci[oó]n\s+trimestral", re.IGNORECASE)
    VALUE_DESEST_RE = re.compile(
        r"\b(valor|cu[aá]nto|cu[aá]nta|nivel|cifra|dato|cu[aá]l\s+es)\b.{0,40}"
        r"\b(desestacionaliz\w*|sin\s+estacionalidad|"
        r"ajustad[oa]\s+por\s+estacionalidad)\b"
        r"|"
        r"\b(desestacionaliz\w*|sin\s+estacionalidad)\b.{0,40}"
        r"\b(valor|cu[aá]nto|cu[aá]nta|nivel|cifra|dato)\b",
        re.IGNORECASE,
    )

    # ---- LOGICA GENERAL ----------------------------------------------
    @classmethod
    def natural_freq_is_prev_period(cls, ent: ResolvedEntities) -> None:
        q = ent.question or ""
        if not q or cls.YOY_KEYWORDS_RE.search(q):
            return
        if str(ent.calc_mode_cls or "").strip().lower() in (
            "contribution", "share", "prev_period"
        ):
            return

        matched_indicator = cls._infer_indicator_from_verb(q, ent.indicator_ent)
        if matched_indicator is None:
            return

        ent.indicator_ent = matched_indicator
        ent.calc_mode_cls = "prev_period"
        ent.seasonality_ent = "sa"
        _trace(ent, cls.CAT, "natural_freq_is_prev_period",
               f"variación natural {matched_indicator} → prev_period+sa")

    @classmethod
    def sa_implies_prev_period(cls, ent: ResolvedEntities) -> None:
        if str(ent.seasonality_ent or "").strip().lower() != "sa":
            return
        if str(ent.calc_mode_cls or "").strip().lower() not in ("", "original"):
            return
        if cls.YOY_KEYWORDS_RE.search(ent.question or ""):
            return
        ent.calc_mode_cls = "prev_period"
        _trace(ent, cls.CAT, "sa_implies_prev_period",
               "seasonality=sa + ambiguo → prev_period")

    # ---- Methods ------------------------------------------------------
    @classmethod
    def _infer_indicator_from_verb(
        cls, question: str, current_indicator: Optional[str]
    ) -> Optional[str]:
        indicator = str(current_indicator or "").strip().lower()
        if indicator == "imacec" and cls.VAR_MENSUAL_RE.search(question):
            return "imacec"
        if indicator == "pib" and cls.VAR_TRIMESTRAL_RE.search(question):
            return "pib"
        if not indicator:
            if cls.VAR_MENSUAL_RE.search(question):
                return "imacec"
            if cls.VAR_TRIMESTRAL_RE.search(question):
                return "pib"
        return None

    # ---- Functions (ruteo) -------------------------------------------
    @classmethod
    def is_value_desestacionalizado(cls, question: str) -> bool:
        """Override de ruteo (ingest.py): True si la intención es pedir el
        valor de la serie desestacionalizada."""
        return bool(question) and bool(cls.VALUE_DESEST_RE.search(question))

    # Exceptions: si la pregunta contiene yoy explícito, no se aplica.


# ============================================================================
# TIPO DE CONSULTA: 03 — VALIDACIÓN DE PERÍODO (IMACEC)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   IMACEC sólo se publica con frecuencia mensual. Si el clasificador asignó
#   otra frecuencia (anual / trimestral) cuando indicator='imacec', se fuerza
#   frequency_ent='m'.
#
# Input   : ent.indicator_ent='imacec', ent.frequency_ent != 'm'
# Output  : ent.frequency_ent='m'
# ============================================================================


class Rule03_ValidacionPeriodoIMACEC:
    """REGLA_03_VALIDACION_PERIODO_IMACEC."""

    CAT = "CAT03"

    @classmethod
    def force_monthly(cls, ent: ResolvedEntities) -> None:
        if not cls._is_imacec_non_monthly(ent):
            return
        ent.frequency_ent = "m"
        _trace(ent, cls.CAT, "imacec_force_monthly", "frequency=m")

    @staticmethod
    def _is_imacec_non_monthly(ent: ResolvedEntities) -> bool:
        return (
            str(ent.indicator_ent or "").strip().lower() == "imacec"
            and str(ent.frequency_ent or "").strip().lower() != "m"
        )

    # Functions / Exceptions / Regex: (none)


# ============================================================================
# TIPO DE CONSULTA: 04 — CRECIMIENTO PIB (no mensual)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   El PIB no se publica mensualmente. Si la consulta llega con PIB y
#   frecuencia mensual, se redirige a trimestral, se descarta el período
#   solicitado (mes específico) y se marca req_form='latest'.
#
# Input   : ent.indicator_ent='pib', ent.frequency_ent='m'
# Output  : ent.frequency_ent='q', ent.period_ent=[], ent.req_form_cls='latest'
# ============================================================================


class Rule04_CrecimientoPIB:
    """REGLA_04_CRECIMIENTO_PIB."""

    CAT = "CAT04"

    @classmethod
    def redirect_pib_monthly_to_quarterly(cls, ent: ResolvedEntities) -> None:
        if not cls._is_pib_monthly(ent):
            return
        ent.frequency_ent = "q"
        ent.req_form_cls = "latest"
        ent.period_ent = []
        _trace(ent, cls.CAT, "redirect_pib_monthly_to_quarterly",
               "PIB mensual → trimestral")

    @staticmethod
    def _is_pib_monthly(ent: ResolvedEntities) -> bool:
        return (
            ent.indicator_ent == "pib"
            and str(ent.frequency_ent or "").strip().lower() == "m"
        )

    # Functions / Exceptions / Regex: (none)


# ============================================================================
# TIPO DE CONSULTA: 05 — FUERA DE ALCANCE
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Bloquea preguntas que no son sobre PIB/IMACEC. Esta regla es de RUTEO
#   (la consume ingest.py); no muta ``ResolvedEntities``.
#
#   Reglas en orden:
#     1. Calendario PIB/IMACEC → en alcance.
#     2. NER detectó activity/region/investment → en alcance.
#     3. NER detectó person/organization/location → fuera de alcance.
#     4. Followup con indicador previo PIB/IMACEC → en alcance.
#     5. Allowlist textual (pib/imacec/...) → en alcance.
#     6. indicator normalizado en {pib, imacec} → en alcance.
#     7. Caso contrario → fuera de alcance.
#
# Input   : question, current_norm, prev_indicator, context_label
# Output  : bool (True = bloquear)
# ============================================================================


class Rule05_FueraDeAlcance:
    """REGLA_05_FUERA_DE_ALCANCE."""

    CAT = "CAT05"

    # ---- Regex --------------------------------------------------------
    ALLOWLIST_RE = re.compile(
        r"\b("
        r"pib|imacec|"
        r"producto\s+interno\s+bruto|"
        r"cuentas?\s+nacional(?:es)?|"
        r"actividad\s+econ[oó]mica|"
        r"econom[ií]a"
        r")\b"
        r"|"
        # Ranking regional implícito: "región que más/menos creció/crecimiento"
        # No menciona PIB/IMACEC explícitamente pero es una consulta de datos macro.
        r"(?=.*\bregi[oó]n\b)(?=.*\b(m[aá]s|menos|mayor|menor)\b)(?=.*\bcreci)",
        re.IGNORECASE,
    )

    # Blocklist textual: cualquier mención clara a tópicos fuera de
    # PIB/IMACEC fuerza out-of-scope incluso si el clasificador
    # devolvió indicator=imacec por defecto. Tiene PRECEDENCIA sobre
    # la allowlist y sobre el follow-up macro: si el usuario pregunta
    # explícitamente por dólar/paridades/tasa/persona, la respuesta
    # debe ser scope_block.
    BLOCKLIST_RE = re.compile(
        r"\b("
        r"paridad(?:es)?|"
        r"d[oó]lar(?:es)?|euro|yen|yuan|libra\s+esterlina|"
        r"peso\s+(?:chileno|argentino|colombiano|mexicano|uruguayo|peruano)|"
        r"tipos?\s+de\s+cambio|tcn|tcr|tcm|"
        r"tasa(?:s)?\s+(?:de\s+)?(?:inter[eé]s|pol[ií]tica|tpm)|tpm|"
        r"inflaci[oó]n|ipc|ipom|ipsa|"
        r"presidenta?|presidente|vicepresidente|gerente|ministr[oa]|director(?:a)?|consejer[oa]|"
        r"qui[eé]n(?:es)?\s+(?:es|son|fue|fueron)|c[oó]mo\s+se\s+llama"
        r")\b",
        re.IGNORECASE,
    )

    # ---- LOGICA GENERAL ----------------------------------------------
    @classmethod
    def is_out_of_scope(
        cls,
        question: str,
        current_norm: Dict[str, Any],
        prev_indicator: Any,
        context_label: str,
    ) -> bool:
        q = _ensure_text(question)
        if not q.strip():
            return False
        if Rule06_Calendario.is_calendar_intent(q):
            return False

        norm = current_norm if isinstance(current_norm, dict) else {}

        # Hard blocklist: precede a cualquier allowlist/follow-up para
        # neutralizar el sesgo del clasificador (que tiende a etiquetar
        # como IMACEC textos desconocidos como paridades, dólar, etc.).
        if cls.BLOCKLIST_RE.search(q):
            return True

        if cls._has_macro_entity(norm):
            return False
        if cls._has_proper_noun(norm):
            return True
        if cls._is_followup_macro(prev_indicator, context_label):
            return False
        if cls.ALLOWLIST_RE.search(q):
            return False
        return cls._indicator_not_allowed(norm)

    # ---- Methods ------------------------------------------------------
    @staticmethod
    def _has_macro_entity(norm: Dict[str, Any]) -> bool:
        return any(
            not _is_empty_value(norm.get(k))
            for k in ("activity", "region", "investment")
        )

    @staticmethod
    def _has_proper_noun(norm: Dict[str, Any]) -> bool:
        return any(
            not _is_empty_value(norm.get(ent))
            for ent in ("person", "organization", "location", "per", "org", "loc")
        )

    @staticmethod
    def _is_followup_macro(prev_indicator: Any, context_label: str) -> bool:
        if str(context_label or "").strip().lower() != "followup":
            return False
        return str(prev_indicator or "").strip().lower() in ("pib", "imacec")

    @staticmethod
    def _indicator_not_allowed(norm: Dict[str, Any]) -> bool:
        """True (= fuera de alcance) si hay indicator y no es pib/imacec.
        Si no hay indicator, también queda fuera (no matchea nada)."""
        indicator = norm.get("indicator")
        if not indicator:
            return True
        allowed = {"pib", "imacec"}
        if isinstance(indicator, (list, tuple, set)):
            return not any(str(x).strip().lower() in allowed for x in indicator)
        return str(indicator).strip().lower() not in allowed

    # Functions / Exceptions: (none)


# ============================================================================
# TIPO DE CONSULTA: 06 — CALENDARIO DE PUBLICACIÓN
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Detecta intención de calendario/fechas de publicación. Esta regla es de
#   RUTEO (la consume ingest.py). Cuando matchea, ingest fuerza intent='method'
#   para enviar la consulta al RAG de calendario.
#
# Input   : question
# Output  : bool (True = es intención de calendario)
# ============================================================================


class Rule06_Calendario:
    """REGLA_06_CALENDARIO."""

    CAT = "CAT06"

    # ---- Regex --------------------------------------------------------
    INTENT_RE = re.compile(
        r"\bcalendario\b|"
        r"cu[aá]ndo\s+se\s+publica|"
        r"cu[aá]ndo\s+sale|"
        r"cu[aá]ndo\s+publican|"
        r"pr[oó]xim[oa]\s+publicaci[oó]n|"
        r"fecha.*publicaci[oó]n|"
        r"pr[oó]xim[oa]\s+(imacec|pib|cuentas\s+nacionales)",
        re.IGNORECASE,
    )

    # ---- LOGICA GENERAL ----------------------------------------------
    @classmethod
    def is_calendar_intent(cls, question: str) -> bool:
        return bool(question) and bool(cls.INTENT_RE.search(question))

    # Methods / Functions / Exceptions: (none)


# ============================================================================
# TIPO DE CONSULTA: 07 — CONTRIBUCIÓN INDIVIDUAL (default activity IMACEC)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Si el indicador es IMACEC y no se especificó actividad, se asigna
#   activity_ent='imacec' como default para que el lookup del catálogo
#   resuelva la serie agregada.
#
# Input   : ent.indicator_ent='imacec', ent.activity_ent is None
# Output  : ent.activity_ent='imacec'
# ============================================================================


class Rule07_ContribucionIndividual:
    """REGLA_07_CONTRIBUCION_INDIVIDUAL."""

    CAT = "CAT07"

    @classmethod
    def default_activity(cls, ent: ResolvedEntities) -> None:
        if not cls._needs_default(ent):
            return
        ent.activity_ent = "imacec"
        _trace(ent, cls.CAT, "imacec_default_activity", "activity=imacec")

    @staticmethod
    def _needs_default(ent: ResolvedEntities) -> bool:
        return ent.indicator_ent == "imacec" and ent.activity_ent is None

    # Functions / Exceptions / Regex: (none)


# ============================================================================
# TIPO DE CONSULTA: 09 — NIVELES / NOMINALES (precio)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Determina el parámetro de precio para la búsqueda de series con
#   precedencia explícita:
#   1) Si el clasificador detectó precio explícito ('enc' o 'co'), se respeta.
#   2) Si la consulta es nominal/precios corrientes, forzar 'co'.
#   3) Si la consulta pide PIB real, tratarla bajo política de precios
#      corrientes y priorizar nivel original.
#   4) Si la consulta pide nivel explícito, por defecto responder en 'co'.
#   5) Fallback general: 'enc'.
#
# Input   : ent.price_ent, ent.question
# Output  : ent.price, ent.calc_mode_cls (en consultas PIB real)
# ============================================================================


class Rule09_NivelesNominales:
    """REGLA_09_NIVELES_NOMINALES."""

    CAT = "CAT09"

    LEVEL_RE = re.compile(
        r"\bnivel(?:es)?\s+de(?:l|la)?\s+(?!variaci|crecimient|ca[ií]da|aceleraci|alza|subid|bajad)\w+",
        re.IGNORECASE,
    )
    NOMINAL_RE = re.compile(
        r"\bnominal(?:es)?\b|\bprecios?\s+corrientes?\b",
        re.IGNORECASE,
    )
    PIB_REAL_RE = re.compile(
        r"\bpib\b.*\breal(?:es)?\b|\breal(?:es)?\b.*\bpib\b",
        re.IGNORECASE,
    )
    ECONOMY_GROWTH_RE = re.compile(
        r"\b(crec\w*|variaci[oó]n|cay[oó]|ca[ií]da|aument[oó]|expand[ií]\w*|expansi[oó]n)\b"
        r".*\b(econom[ií]a|actividad\s+econ[oó]mica)\b"
        r"|\b(econom[ií]a|actividad\s+econ[oó]mica)\b"
        r".*\b(crec\w*|variaci[oó]n|cay[oó]|ca[ií]da|aument[oó]|expand[ií]\w*|expansi[oó]n)\b",
        re.IGNORECASE,
    )
    YEAR_WORD_RE = re.compile(r"\ba[nñ]os?\b", re.IGNORECASE)

    @classmethod
    def assign_price(cls, ent: ResolvedEntities) -> None:
        q = _ensure_text(ent.question).strip().lower()

        # Desambiguación anual: "crecimiento de la economía" con mención de
        # años (explícita o relativa) siempre debe resolver a PIB, no IMACEC.
        if cls.ECONOMY_GROWTH_RE.search(q) and cls.YEAR_WORD_RE.search(q):
            ent.indicator_ent = "pib"
            ent.frequency_ent = "a"
            if str(ent.activity_ent or "").strip().lower() == "imacec":
                ent.activity_ent = None
            _trace(
                ent,
                cls.CAT,
                "economy_growth_years_disambiguation",
                "indicator=pib, frequency=a",
            )

        # Desambiguación: si "pib real" quedó como IMACEC, corregir a PIB.
        if cls.PIB_REAL_RE.search(q) and str(ent.indicator_ent or "").strip().lower() == "imacec":
            ent.indicator_ent = "pib"
            if str(ent.frequency_ent or "").strip().lower() == "m":
                ent.frequency_ent = "a" if cls.YEAR_WORD_RE.search(q) else "q"
            _trace(
                ent,
                cls.CAT,
                "pib_real_disambiguation",
                f"indicator={ent.indicator_ent}, frequency={ent.frequency_ent}",
            )

        # 1) Respetar precio explícito detectado por normalizador/clasificador.
        if ent.price_ent in ("co", "enc"):
            ent.price = ent.price_ent
            _trace(ent, cls.CAT, "assign_price", f"price={ent.price} (from price_ent)")
            return

        # 2) Nominal / precios corrientes y consultas "PIB real".
        if cls.NOMINAL_RE.search(q) or cls.PIB_REAL_RE.search(q):
            ent.price = "co"
            ent.price_ent = "co"
            if cls.PIB_REAL_RE.search(q):
                ent.calc_mode_cls = "original"
                if "nivel" not in q:
                    ent.question = f"{_ensure_text(ent.question)} (nivel en pesos corrientes)"
                ent.historical_floor_instruction = (
                    "REGLA DE SALIDA (PIB REAL): para esta consulta debes reportar "
                    "el nivel original (campo value) del PIB a precios corrientes. "
                    "NO uses pct ni yoy_pct como cifra principal y NO redactes la "
                    "respuesta como variación porcentual."
                )
                _trace(ent, cls.CAT, "assign_price", "price=co + calc_mode=original (pib real query)")
            else:
                _trace(ent, cls.CAT, "assign_price", "price=co (nominal query)")
            return

        # 3) Consulta de nivel explícita.
        if cls.LEVEL_RE.search(q):
            ent.price = "co"
            _trace(ent, cls.CAT, "assign_price", "price=co (default level query)")
            return

        # 4) Fallback general.
        ent.price = "enc"
        _trace(ent, cls.CAT, "assign_price", f"price={ent.price} (default)")

    # Methods / Functions / Exceptions / Regex: (none)


# ============================================================================
# TIPO DE CONSULTA: 10 — PARTICIPACIONES (share)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Detecta intención de participación a partir del texto ("cuánto pesa",
#   "qué porcentaje", "participación", "peso del/en") cuando hay contexto
#   de inversión PIB. Redirige a calc_mode='share', frequency='a',
#   price='co'.
#
# Input   : ent.question, ent.indicator_ent, ent.investment_cls/ent
# Output  : ent.calc_mode_cls='share', ent.frequency_ent='a', ent.price_ent='co'
#           ent.investment_cls='general' si era 'none'.
# ============================================================================


class Rule10_Participaciones:
    """REGLA_10_PARTICIPACIONES."""

    CAT = "CAT10"

    # ---- Regex --------------------------------------------------------
    INTENT_RE = re.compile(
        r"(?:cu[aá]nto\s+pesa|qu[eé]\s+porcentaje|participaci[oó]n|"
        r"peso\s+(?:del?|en))",
        re.IGNORECASE,
    )

    # ---- LOGICA GENERAL ----------------------------------------------
    @classmethod
    def detect_share_intent(cls, ent: ResolvedEntities) -> None:
        if not cls._matches(ent):
            return
        ent.calc_mode_cls = "share"
        ent.frequency_ent = "a"
        ent.price_ent = "co"
        if ent.investment_cls == "none":
            ent.investment_cls = "general"
        _trace(ent, cls.CAT, "detect_share_intent",
               "share + freq=a + price=co")

    # ---- Methods ------------------------------------------------------
    @classmethod
    def _matches(cls, ent: ResolvedEntities) -> bool:
        q = str(ent.question or "").strip()
        if not q or not cls.INTENT_RE.search(q):
            return False
        if str(ent.indicator_ent or "").strip().lower() not in ("pib", ""):
            return False
        has_inv = (
            ent.investment_cls in ("specific", "general")
            or ent.investment_ent is not None
        )
        return has_inv

    # Functions / Exceptions: (none)


# ============================================================================
# TIPO DE CONSULTA: 13 — DATOS HISTÓRICOS (pisos por indicador)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Aplica pisos históricos:
#     - IMACEC: piso 1996 (datos empalmados).
#     - PIB:    piso 1960; flag hist=1 si año < 1996.
#   Si la consulta solicita un año previo al piso, se reescribe el período
#   y se genera ``historical_floor_instruction`` para que el LLM lo informe.
#
# Input   : ent.indicator_ent, ent.period_ent
# Output  : ent.hist, ent.period_ent, ent.historical_floor_instruction
# ============================================================================


class Rule13_Historicos:
    """REGLA_13_HISTORICOS."""

    CAT = "CAT13"

    PIB_FLOOR_YEAR = 1960
    IMACEC_FLOOR_YEAR = 1996
    HIST_FLAG_YEAR = 1996

    # ---- Regex (BUG-A: detección de intención "primer dato histórico") --------
    EARLIEST_RE = re.compile(
        r"\bm[aá]s\s+antiguo\b"
        r"|\bm[aá]s\s+viejo\b"
        r"|\bprimer\s+(?:dato|valor|registro|publicacion|imacec|pib)\b"
        r"|\bprimero\s+(?:disponible|publicado|registrado)\b"
        r"|\bhistor\w+\s+disponible\s+m[aá]s\s+antiguo\b"
        r"|\bdesde\s+cu[aá]ndo\b",
        re.IGNORECASE,
    )
    # Guard: "primer trimestre" / "primera región" NO activan earliest
    EARLIEST_GUARD_RE = re.compile(r"\btrimestre\b|\bregio\w*\b", re.IGNORECASE)

    # ---- LOGICA GENERAL ----------------------------------------------
    @classmethod
    def apply_floor(cls, ent: ResolvedEntities) -> None:
        # BUG-A: detectar intent de dato más antiguo disponible
        q = ent.question or ""
        if cls.EARLIEST_RE.search(q) and not cls.EARLIEST_GUARD_RE.search(q):
            ent.date_direction = "earliest"
            _trace(ent, cls.CAT, "earliest_intent",
                   "date_direction=earliest (primer dato histórico)")
            return

        indicator = str(ent.indicator_ent or "").strip().lower()
        period = list(ent.period_ent or [])
        ref_year = _extract_year(period[0]) if period else None

        if indicator == "imacec":
            cls._apply_imacec(ent, period, ref_year)
            return
        cls._apply_pib_or_other(ent, indicator, period, ref_year)

    # ---- Methods ------------------------------------------------------
    @classmethod
    def _apply_imacec(
        cls, ent: ResolvedEntities, period: List[Any], ref_year: Optional[int]
    ) -> None:
        ent.hist = 0
        if ref_year is None or ref_year >= cls.IMACEC_FLOOR_YEAR:
            return
        ent.period_ent = cls._rewrite_period_to_floor(period, cls.IMACEC_FLOOR_YEAR)
        ent.historical_floor_instruction = (
            "REGLA DE DISPONIBILIDAD HISTÓRICA (IMACEC): solo hay datos "
            "empalmados de IMACEC desde 1996. Debes indicarlo explícitamente "
            "y reportar 1996 (o rango desde 1996, según corresponda)."
        )
        _trace(ent, cls.CAT, "pib_hist_flag", "IMACEC piso 1996")

    @classmethod
    def _apply_pib_or_other(
        cls,
        ent: ResolvedEntities,
        indicator: str,
        period: List[Any],
        ref_year: Optional[int],
    ) -> None:
        # BUG-B1: PIB Regional no tiene cuadro histórico separado con hist=1;
        # forzar hist=0 para que el buscador use el cuadro regional estándar.
        if str(ent.region_cls or "").strip().lower() == "specific":
            ent.hist = 0
            _trace(ent, cls.CAT, "pib_regional_no_hist_flag",
                   "region_cls=specific → hist=0 (no cuadro hist PIB regional)")
            return

        ent.hist = 1 if (ref_year is not None and ref_year < cls.HIST_FLAG_YEAR) else 0
        if indicator != "pib" or ref_year is None or ref_year >= cls.PIB_FLOOR_YEAR:
            return
        ent.period_ent = cls._rewrite_period_to_floor(period, cls.PIB_FLOOR_YEAR)
        ent.historical_floor_instruction = (
            "REGLA DE DISPONIBILIDAD HISTÓRICA (PIB): hay datos empalmados "
            "desde 1960. Debes indicarlo explícitamente y reportar 1960 "
            "(o rango desde 1960, según corresponda)."
        )
        adjusted_year = _extract_year(ent.period_ent[0]) if ent.period_ent else None
        ent.hist = 1 if (
            adjusted_year is not None and adjusted_year < cls.HIST_FLAG_YEAR
        ) else 0
        _trace(ent, cls.CAT, "pib_hist_flag", "PIB piso 1960")

    # ---- Functions ----------------------------------------------------
    @classmethod
    def _rewrite_period_to_floor(
        cls, period_ent: List[Any], floor_year: int
    ) -> List[Any]:
        if not period_ent:
            return []
        if len(period_ent) == 1:
            return [cls._replace_year_token(period_ent[0], floor_year)]
        start = cls._replace_year_token(period_ent[0], floor_year)
        end_year = _extract_year(period_ent[-1])
        end = (
            cls._replace_year_token(period_ent[-1], floor_year)
            if (end_year is not None and end_year < floor_year)
            else period_ent[-1]
        )
        return [start, end]

    @staticmethod
    def _replace_year_token(value: Any, year: int) -> str:
        text = str(value or "").strip()
        if not text:
            return str(year)
        if re.fullmatch(r"(19|20)\d{2}", text):
            return str(year)
        return re.sub(r"(19|20)\d{2}", str(year), text, count=1)

    # Exceptions / Regex: (none)


# ============================================================================
# TIPO DE CONSULTA: 14 — PIB REGIONAL · default YoY
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Para PIB regional la convención canónica del BCCh es reportar la
#   variación interanual (YoY), que es la vista por defecto del cuadro
#   Cuentas Nacionales > PIB Regional. Si el clasificador no detectó
#   intención variacional explícita (queda ``calc_mode=original``) y la
#   pregunta es puntual (``req_form=point``), forzamos
#   ``calc_mode_cls='yoy'`` para que:
#     1. ``response.py::_is_level_only_query`` NO tome la ruta de NIVEL.
#     2. El URL builder use ``cbCalculo=YTYPCT`` (consistente con la imagen
#        del cuadro BDE).
#     3. El LLM reporte ``yoy_pct`` y no el monto en miles de millones.
#
#   Guardas (todas en AND para evitar afectar otras consultas):
#     - indicator_ent == 'pib'
#     - region_cls == 'specific' AND region_ent normalizada
#     - calc_mode_cls == 'original' (sin señal variacional del clasificador)
#     - req_form_cls == 'point'
#     - sin señal léxica de "nivel/monto/valor en pesos" en la pregunta
#
#   Si el usuario pide explícitamente el nivel ("monto", "valor en pesos",
#   "miles de millones", "cuántos pesos"), la regla NO se activa y se
#   conserva ``calc_mode=original``.
#
# Input   : ent.indicator_ent, ent.region_cls, ent.region_ent,
#           ent.calc_mode_cls, ent.req_form_cls, ent.question
# Output  : ent.calc_mode_cls = 'yoy'  (cuando aplica)
# ============================================================================


class Rule14_PibRegionalDefaultYoY:
    """REGLA_14_PIB_REGIONAL_DEFAULT_YOY."""

    CAT = "CAT14"

    # Alias para compatibilidad: la fuente de verdad es ``_LEVEL_HINTS_RE``.
    LEVEL_HINTS_RE = _LEVEL_HINTS_RE

    @classmethod
    def force_yoy(cls, ent: ResolvedEntities) -> None:
        if (str(ent.indicator_ent or "").strip().lower()) != "pib":
            return
        if (str(ent.region_cls or "").strip().lower()) != "specific":
            return
        if not (str(ent.region_ent or "").strip()):
            return
        if (str(ent.calc_mode_cls or "").strip().lower()) != "original":
            return
        # Aplica tanto a req_form='point' (período explícito) como a
        # req_form='latest' (consulta genérica "cuál es el pib de la región X").
        if (str(ent.req_form_cls or "").strip().lower()) not in ("point", "latest"):
            return
        q = _ensure_text(ent.question)
        if cls.LEVEL_HINTS_RE.search(q):
            return
        ent.calc_mode_cls = "yoy"
        _trace(
            ent,
            cls.CAT,
            "pib_regional_default_yoy",
            f"region_ent={ent.region_ent} → calc_mode=yoy",
        )

    # Methods / Functions / Exceptions: (none)


# ============================================================================
# TIPO DE CONSULTA: 12 — PIB REGIONAL · clarificaciones geográficas (BUG-B2)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   "La Antártica" no existe como región independiente en el PIB Regional.
#   Es parte de la Región XII: "Magallanes y de la Antártica Chilena".
#   El normalizer mapea "antartica" → "magallanes" (correcto), pero cuando
#   el usuario dice "antártica" SIN mencionar "magallanes", es necesario
#   aclarar la situación en la respuesta para no devolver datos silenciosamente.
#   Se detecta aquí y se propaga via region_alias_note al data node (response.py).
#
# Input   : ent.region_ent, ent.question
# Output  : ent.region_alias_note = "antartica_standalone"
# ============================================================================


class Rule12_PibRegional:
    """REGLA_12_PIB_REGIONAL — clarificaciones geográficas."""

    CAT = "CAT12"

    # ---- Regex --------------------------------------------------------
    ANTARTICA_RE  = re.compile(r"\bantar[ct]ica\b", re.IGNORECASE)
    MAGALLANES_RE = re.compile(r"\bmagallanes\b",   re.IGNORECASE)

    # ---- LOGICA GENERAL ----------------------------------------------
    @classmethod
    def flag_antartica_standalone(cls, ent: ResolvedEntities) -> None:
        """Marca consultas donde el usuario dijo 'antártica' sin 'magallanes'."""
        if str(ent.region_ent or "").strip().lower() != "magallanes":
            return
        q = ent.question or ""
        if cls.ANTARTICA_RE.search(q) and not cls.MAGALLANES_RE.search(q):
            ent.region_alias_note = "antartica_standalone"
            _trace(ent, cls.CAT, "antartica_standalone",
                   "region_alias_note=antartica_standalone")

    # Methods / Functions / Exceptions / Regex: (none)


# ============================================================================
# TIPO DE CONSULTA: 15 — CRECIMIENTO + AÑOS DE CHILE → PIB (no IMACEC)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Cuando el usuario pregunta por el "crecimiento" de Chile en un horizonte
#   de años ("últimos 10 años", "últimos años", "de los años"), el indicador
#   correcto es el PIB (no el IMACEC). El normalizer defaultea a IMACEC
#   cuando no se menciona un indicador explícito; esta regla corrige ese
#   default a PIB y deja que Rule16 setee la frecuencia anual.
#
# Input   : ent.question, ent.indicator_ent='imacec' (default), sin actividad
# Output  : ent.indicator_ent='pib', ent.frequency_ent='a' (si no se setea por Rule16)
# ============================================================================


class Rule15_CrecimientoChileToPIB:
    """REGLA_15_CRECIMIENTO_CHILE_PIB."""

    CAT = "CAT15"

    # "crecimiento" o "crecio"/"crece" como rasgo del verbo principal
    CRECIMIENTO_RE = re.compile(
        r"\bcrecimiento\b|\bcreci[oó]\b|\bcrece\b|\bcrecen\b",
        re.IGNORECASE,
    )
    # Horizonte plurianual: "años", "últimos N años", "de los últimos años"
    ANIOS_RE = re.compile(r"\baños?\b", re.IGNORECASE)
    # Guard: si menciona "mes", "trimestre" o un mes concreto, no aplica.
    GUARD_RE = re.compile(
        r"\bmes(?:es|ual)?\b|\btrimestr\w*\b|"
        r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|setiembre|octubre|noviembre|diciembre)\b",
        re.IGNORECASE,
    )

    @classmethod
    def force_pib_for_crecimiento_anual(cls, ent: ResolvedEntities) -> None:
        q = _ensure_text(ent.question)
        if not q:
            return
        if not cls.CRECIMIENTO_RE.search(q):
            return
        if not cls.ANIOS_RE.search(q):
            return
        if cls.GUARD_RE.search(q):
            return
        # Solo overridear cuando el indicador es imacec (default del normalizer)
        # y no hay actividad específica (las actividades del IMACEC — ej. minería —
        # deben mantenerse en IMACEC).
        if str(ent.indicator_ent or "").strip().lower() != "imacec":
            return
        if str(ent.activity_ent or "").strip():
            return
        ent.indicator_ent = "pib"
        # Rule16 seteará freq='a' si corresponde; aquí solo aseguramos que la
        # frecuencia mensual heredada del default IMACEC no quede colgada.
        if str(ent.frequency_ent or "").strip().lower() == "m":
            ent.frequency_ent = "a"
        _trace(
            ent,
            cls.CAT,
            "crecimiento_anios_to_pib",
            "indicator_ent=imacec→pib por 'crecimiento'+'años'",
        )


# ============================================================================
# TIPO DE CONSULTA: 16 — PIB CON HORIZONTE DE AÑOS → FRECUENCIA ANUAL
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   "PIB de los últimos N años" / "PIB en los años ..." / "PIB anual".
#   Cuando el usuario menciona un horizonte explícito en años y NO menciona
#   trimestre/mes, la frecuencia natural es anual (no trimestral por default
#   del normalizer). También limpia el período si el normalizer lo resolvió
#   incorrectamente a un solo trimestre ("últimos 10" → [2026-Q1] es bug).
#
# Input   : ent.indicator_ent='pib', question con "\d+ años" o "últimos años"
# Output  : ent.frequency_ent='a'; ent.period_ent=[] cuando el rango
#           tiene >1 año y el período es de un solo trimestre.
# ============================================================================


class Rule16_PibLargoPlazoAnual:
    """REGLA_16_PIB_LARGO_PLAZO_ANUAL."""

    CAT = "CAT16"

    # Detecta horizonte plurianual: "10 años", "últimos 10 años", "últimos años",
    # "varios años", "última década". También captura "de los últimos años".
    HORIZON_RE = re.compile(
        r"\b(?:\d+|varios|\u00faltim[oa]s?|ultim[oa]s?|pasad[oa]s?|recientes?)\s+años?\b"
        r"|\b\u00faltim[oa]\s+d[eé]cada\b"
        r"|\bultima\s+decada\b",
        re.IGNORECASE,
    )
    # Guard: si menciona "trimestre" o "mes" explícitamente, no forzar anual.
    GUARD_RE = re.compile(
        r"\btrimestr\w*\b|\bmes(?:es|ual)?\b",
        re.IGNORECASE,
    )
    # Detecta número de años en el horizonte (para limpiar período bugueado)
    N_ANIOS_RE = re.compile(r"\b(\d+)\s+años?\b", re.IGNORECASE)

    @classmethod
    def force_annual_frequency(cls, ent: ResolvedEntities) -> None:
        if str(ent.indicator_ent or "").strip().lower() != "pib":
            return
        q = _ensure_text(ent.question)
        if not q:
            return
        if not cls.HORIZON_RE.search(q):
            return
        if cls.GUARD_RE.search(q):
            return
        ent.frequency_ent = "a"
        # Limpia período bugueado: el normalizer suele resolver "últimos 10"
        # como el trimestre actual (un solo trimestre). Si el rango es
        # plurianual pero period_ent cubre <2 años, lo descartamos para que
        # el data_node elija el rango por default (toda la serie anual).
        m = cls.N_ANIOS_RE.search(q)
        n_years = int(m.group(1)) if m else 0
        period = list(ent.period_ent or [])
        if period:
            years = {y for y in (_extract_year(p) for p in period) if y is not None}
            span = (max(years) - min(years) + 1) if years else 0
            if n_years >= 2 and span < n_years:
                ent.period_ent = []
        _trace(
            ent,
            cls.CAT,
            "pib_anual_largo_plazo",
            f"frequency_ent=a (horizonte plurianual, n_anios~{n_years})",
        )


# ============================================================================
# TIPO DE CONSULTA: 17 — PIB REGIONAL RANKING (más/menos creció)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Consultas como "cuál es la región que más/menos creció el último trimestre"
#   o "qué región creció más/menos" no nombran una región específica, pero sí
#   implican el contexto de PIB regional. El clasificador a veces no detecta
#   region_cls="general" para estas preguntas (especialmente con "menos creció"),
#   lo que resulta en has_region=0 y ruteo al cuadro de PIB nacional.
#   Esta regla fuerza region_cls="general", indicator_ent="pib" y
#   calc_mode_cls="yoy" para que el catalog devuelva el cuadro regional.
#
# Input   : ent.question con patrón "regi[oó]n" + "más/menos creció"
# Output  : ent.region_cls="general", ent.indicator_ent="pib",
#           ent.calc_mode_cls="yoy"
# ============================================================================


class Rule17_PibRegionalRanking:
    """REGLA_17_PIB_REGIONAL_RANKING — región que más/menos creció."""

    CAT = "CAT17"

    # Pregunta que menciona "región" + comparativo de crecimiento (cualquier orden relativo)
    # Cubre: "región que más/menos creció", "región creció más/menos", etc.
    REGION_RE = re.compile(r"\bregi[oó]n\b", re.IGNORECASE)
    CRECI_RE = re.compile(r"\bcreci(miento|[oó]|[eé]|[eé]n|endo)?\b", re.IGNORECASE)
    COMP_RE = re.compile(r"\b(m[aá]s|menos|mayor(?:es)?|menor(?:es)?)\b", re.IGNORECASE)

    @classmethod
    def force_regional_ranking(cls, ent: ResolvedEntities) -> None:
        # Ya tiene región específica resuelta → no aplica (Rule14 lo maneja)
        if (
            str(ent.region_cls or "").strip().lower() == "specific"
            and str(ent.region_ent or "").strip()
        ):
            return
        q = _ensure_text(ent.question)
        if not (cls.REGION_RE.search(q) and cls.CRECI_RE.search(q) and cls.COMP_RE.search(q)):
            return
        ent.indicator_ent = "pib"
        ent.region_cls = "general"
        if str(ent.calc_mode_cls or "").strip().lower() not in ("yoy", "prev_period"):
            ent.calc_mode_cls = "yoy"
        _trace(
            ent,
            cls.CAT,
            "pib_regional_ranking",
            "region_cls=general, indicator_ent=pib, calc_mode=yoy (ranking más/menos creció)",
        )


# ============================================================================
# TIPO DE CONSULTA: 18 — PIB POR ACTIVIDAD · default YoY (no regional)
# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Análogo a Rule14 pero para PIB por ACTIVIDAD (sin región). Cuando el
#   usuario pregunta "cuál es el pib minero", "cuál es el valor del pib de
#   transporte", etc., el clasificador deja calc_mode=original, lo que hace
#   que el data node retorne el NIVEL en miles de millones de pesos y, peor
#   aún, que la búsqueda en la familia de niveles seleccione una serie
#   distinta a la solicitada (ej. "pib minero" → "pib no minero") porque la
#   organización de series por nivel difiere de la organización YoY.
#
#   La interpretación natural de "cuál es el PIB de <actividad>" es la
#   variación interanual (yoy), salvo que el usuario indique nivel/monto
#   explícitamente (LEVEL_HINTS_RE).
#
# Trigger : indicator_ent=pib, activity_cls=specific, region_cls!=specific,
#           calc_mode_cls=original, req_form_cls in {point, latest},
#           sin hints léxicos de nivel.
# Output  : calc_mode_cls=yoy
# ============================================================================


class Rule18_PibActivityDefaultYoY:
    """REGLA_18_PIB_ACTIVITY_DEFAULT_YOY."""

    CAT = "CAT18"

    # Alias para compatibilidad: la fuente de verdad es ``_LEVEL_HINTS_RE``.
    LEVEL_HINTS_RE = _LEVEL_HINTS_RE

    # req_form aceptados:
    #   - 'point'  → período explícito ("pib minero del 2024")
    #   - 'latest' → consulta genérica ("cuál es el pib minero")
    #   - 'range'  → rango temporal ("pib minero últimos 4 trimestres")
    _REQ_FORMS = ("point", "latest", "range")

    @classmethod
    def force_yoy(cls, ent: ResolvedEntities) -> None:
        if (str(ent.indicator_ent or "").strip().lower()) != "pib":
            return
        # Debe ser actividad específica resuelta
        if (str(ent.activity_cls or "").strip().lower()) != "specific":
            return
        if not (str(ent.activity_ent or "").strip()):
            return
        # NO aplicar si es regional (Rule14 maneja regional)
        if (str(ent.region_cls or "").strip().lower()) == "specific":
            return
        if (str(ent.calc_mode_cls or "").strip().lower()) != "original":
            return
        if (str(ent.req_form_cls or "").strip().lower()) not in cls._REQ_FORMS:
            return
        q = _ensure_text(ent.question)
        if cls.LEVEL_HINTS_RE.search(q):
            return
        ent.calc_mode_cls = "yoy"
        _trace(
            ent,
            cls.CAT,
            "pib_activity_default_yoy",
            f"activity_ent={ent.activity_ent} → calc_mode=yoy",
        )

    # Methods / Functions / Exceptions: (none)


# ============================================================================
#
# EXPLICACIÓN DEL PROCESO
#   Detecta saludos puros ("hola", "buenos días"). Regla de RUTEO consumida
#   por ingest.py para enviar la conversación al greeting_node.
#
# Input   : question
# Output  : bool
# ============================================================================


class RuleGreeting:
    """REGLA_GREETING (saludos)."""

    CAT = "GREET"

    # ---- Regex --------------------------------------------------------
    # Lista de "tokens" de saludo. Una pregunta es saludo si toda ella
    # se compone de uno o más de estos tokens, separados por puntuación
    # o espacios (p.ej. "hola", "como estas", "hola, como estas",
    # "buenos dias que tal", "hola como va?").
    _GREETING_TOKEN_RE = (
        r"hola+|holi+|holaa+|hey|hi|hello|saludos?|"
        r"buen[oa]s?\s*(?:d[ií]as?|tardes?|noches?)?|buen\s*d[ií]a|"
        r"qu[eé]\s*tal|qu[eé]\s+haces|qu[eé]\s+onda|"
        r"c[oó]mo\s+(?:est[aá]s|est[aá]n|est[aá]is|va|vas|andas|te\s+va|estamos)|"
        r"todo\s+bien|y\s+t[uú]|y\s+vos|gracias"
    )
    GREETING_RE = re.compile(
        rf"^\s*(?:(?:{_GREETING_TOKEN_RE})\s*[,\.!\?¡¿]*\s*)+$",
        re.IGNORECASE,
    )

    # ---- LOGICA GENERAL ----------------------------------------------
    @classmethod
    def is_greeting(cls, question: str) -> bool:
        if not question:
            return False
        return bool(cls.GREETING_RE.match(question.strip()))

    # Methods / Functions / Exceptions: (none)


# ============================================================================
# PIPELINE — apply_business_rules
# ============================================================================
#
# Mantiene el ORDEN HISTÓRICO de invocación para preservar el comportamiento
# observado en producción. NO reordenar sin tests de regresión.
# ============================================================================


_RULES_PIPELINE = [
    Rule10_Participaciones.detect_share_intent,
    Rule02_VariacionesDesest.natural_freq_is_prev_period,
    Rule02_VariacionesDesest.sa_implies_prev_period,
    Rule01_ContribucionGrupal.force_general,
    Rule03_ValidacionPeriodoIMACEC.force_monthly,
    Rule15_CrecimientoChileToPIB.force_pib_for_crecimiento_anual,  # antes de Rule07/09
    Rule07_ContribucionIndividual.default_activity,
    Rule09_NivelesNominales.assign_price,
    Rule13_Historicos.apply_floor,
    Rule01_ContribucionGrupal.demanda_interna,
    Rule04_CrecimientoPIB.redirect_pib_monthly_to_quarterly,
    Rule16_PibLargoPlazoAnual.force_annual_frequency,              # después de Rule04
    Rule17_PibRegionalRanking.force_regional_ranking,     # antes de Rule14
    Rule14_PibRegionalDefaultYoY.force_yoy,
    Rule18_PibActivityDefaultYoY.force_yoy,         # PIB por actividad (no regional)
    Rule12_PibRegional.flag_antartica_standalone,   # BUG-B2: alias geográfico
]


def apply_business_rules(ent: ResolvedEntities) -> ResolvedEntities:
    """Ejecuta el pipeline en orden. Errores por regla se loguean y no abortan."""
    for rule in _RULES_PIPELINE:
        try:
            rule(ent)
        except Exception:
            logger.exception("[RULE] error en %s", getattr(rule, "__qualname__", rule))
    return ent


# ============================================================================
# BACKWARD-COMPAT — aliases planos a nivel módulo
# ============================================================================
#
# Estos nombres se mantienen para no romper:
#   - orchestrator.data._business_rules (shim re-exporta de aquí)
#   - orchestrator.graph.nodes.ingest   (usa los predicados _is_*)
# Ante cualquier renombrado, sincronizar con check_files_rules.py.
# ============================================================================

# Regex (alias) ---------------------------------------------------------------
_CALENDAR_INTENT_RE = Rule06_Calendario.INTENT_RE
_VALUE_DESEST_RE = Rule02_VariacionesDesest.VALUE_DESEST_RE
_GREETING_RE = RuleGreeting.GREETING_RE
_OUT_OF_SCOPE_ALLOWLIST_RE = Rule05_FueraDeAlcance.ALLOWLIST_RE
_SHARE_INTENT_RE = Rule10_Participaciones.INTENT_RE
_YOY_KEYWORDS_RE = Rule02_VariacionesDesest.YOY_KEYWORDS_RE
_VAR_MENSUAL_RE = Rule02_VariacionesDesest.VAR_MENSUAL_RE
_VAR_TRIMESTRAL_RE = Rule02_VariacionesDesest.VAR_TRIMESTRAL_RE


# Predicates de ruteo (alias) -------------------------------------------------
def _is_calendar_intent(question: str) -> bool:
    return Rule06_Calendario.is_calendar_intent(question)


def _is_value_desestacionalizado(question: str) -> bool:
    return Rule02_VariacionesDesest.is_value_desestacionalizado(question)


def _is_greeting(question: str) -> bool:
    return RuleGreeting.is_greeting(question)


def _is_out_of_scope(
    question: str,
    current_norm: Dict[str, Any],
    prev_indicator: Any,
    context_label: str,
) -> bool:
    return Rule05_FueraDeAlcance.is_out_of_scope(
        question, current_norm, prev_indicator, context_label
    )


# Reglas de pipeline (alias) --------------------------------------------------
def _rule_detect_share_intent(ent: ResolvedEntities) -> None:
    Rule10_Participaciones.detect_share_intent(ent)


def _rule_natural_freq_variation_is_prev_period(ent: ResolvedEntities) -> None:
    Rule02_VariacionesDesest.natural_freq_is_prev_period(ent)


def _rule_seasonality_sa_implies_prev_period(ent: ResolvedEntities) -> None:
    Rule02_VariacionesDesest.sa_implies_prev_period(ent)


def _rule_contribution_investment_force_general(ent: ResolvedEntities) -> None:
    Rule01_ContribucionGrupal.force_general(ent)


def _rule_contribution_demanda_interna(ent: ResolvedEntities) -> None:
    Rule01_ContribucionGrupal.demanda_interna(ent)


def _rule_imacec_force_monthly(ent: ResolvedEntities) -> None:
    Rule03_ValidacionPeriodoIMACEC.force_monthly(ent)


def _rule_imacec_default_activity(ent: ResolvedEntities) -> None:
    Rule07_ContribucionIndividual.default_activity(ent)


def _rule_assign_price(ent: ResolvedEntities) -> None:
    Rule09_NivelesNominales.assign_price(ent)


def _rule_pib_hist_flag(ent: ResolvedEntities) -> None:
    Rule13_Historicos.apply_floor(ent)


def _rule_redirect_pib_monthly_to_quarterly(ent: ResolvedEntities) -> None:
    Rule04_CrecimientoPIB.redirect_pib_monthly_to_quarterly(ent)


__all__ = [
    # API pública
    "ResolvedEntities",
    "apply_business_rules",
    # Clases por tipo de consulta
    "Rule01_ContribucionGrupal",
    "Rule02_VariacionesDesest",
    "Rule03_ValidacionPeriodoIMACEC",
    "Rule04_CrecimientoPIB",
    "Rule05_FueraDeAlcance",
    "Rule06_Calendario",
    "Rule07_ContribucionIndividual",
    "Rule09_NivelesNominales",
    "Rule10_Participaciones",
    "Rule13_Historicos",
    "Rule12_PibRegional",
    "Rule14_PibRegionalDefaultYoY",
    "Rule15_CrecimientoChileToPIB",
    "Rule16_PibLargoPlazoAnual",
    "Rule17_PibRegionalRanking",
    "Rule18_PibActivityDefaultYoY",
    "RuleGreeting",
    # Predicates de ruteo
    "_is_calendar_intent",
    "_is_value_desestacionalizado",
    "_is_greeting",
    "_is_out_of_scope",
    "_is_empty_cls",
    # Regex compilados
    "_CALENDAR_INTENT_RE",
    "_VALUE_DESEST_RE",
    "_GREETING_RE",
    "_OUT_OF_SCOPE_ALLOWLIST_RE",
    "_SHARE_INTENT_RE",
    "_YOY_KEYWORDS_RE",
    "_VAR_MENSUAL_RE",
    "_VAR_TRIMESTRAL_RE",
    # Hints
    "_PIB_ACTIVITY_HINTS",
    "_IMACEC_ACTIVITY_HINTS",
    "_PREVIOUS_ACTIVITY_HINTS",
    # Reglas (aliases planos)
    "_rule_detect_share_intent",
    "_rule_natural_freq_variation_is_prev_period",
    "_rule_seasonality_sa_implies_prev_period",
    "_rule_contribution_investment_force_general",
    "_rule_contribution_demanda_interna",
    "_rule_imacec_force_monthly",
    "_rule_imacec_default_activity",
    "_rule_assign_price",
    "_rule_pib_hist_flag",
    "_rule_redirect_pib_monthly_to_quarterly",
]
