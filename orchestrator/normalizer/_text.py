"""
Utilidades de texto y matching fuzzy para el normalizador NER.

Funciones de bajo nivel reutilizadas por todos los submódulos:
- Normalización de texto (minúsculas, sin acentos)
- Matching fuzzy con SequenceMatcher
- Búsqueda de mejor clave en vocabularios
- Detección de indicadores genéricos
"""

import re
from typing import Dict, List, Optional
from difflib import SequenceMatcher

from orchestrator.normalizer._vocab import GENERIC_INDICATOR_TERMS


# ─── Normalización de texto ────────────────────────────────────────────────────

_ACCENT_MAP = {
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
    "ü": "u", "ñ": "n",
}


def normalize_text(text: str) -> str:
    """Convierte a minúsculas y elimina acentos.  Ej: ``'Minería'`` → ``'mineria'``."""
    if not text:
        return ""
    text = text.lower().strip()
    for accented, plain in _ACCENT_MAP.items():
        text = text.replace(accented, plain)
    return text


# ─── Fuzzy matching ────────────────────────────────────────────────────────────

def similarity_ratio(a: str, b: str) -> float:
    """Ratio de similitud (0‑1) entre dos strings ya normalizados."""
    return SequenceMatcher(None, a, b).ratio()


def fuzzy_match(
    input_text: str,
    target_terms: List[str],
    threshold: float = 0.75,
) -> Optional[str]:
    """Retorna el término de *target_terms* con mayor similitud si supera *threshold*."""
    if not input_text or not target_terms:
        return None

    input_norm = normalize_text(input_text)
    best, best_ratio = None, 0.0
    for term in target_terms:
        ratio = similarity_ratio(input_norm, normalize_text(term))
        if ratio > best_ratio:
            best_ratio, best = ratio, term
    return best if best_ratio >= threshold else None


# ─── Búsqueda de mejor clave en un vocabulario ────────────────────────────────

def best_vocab_key(
    input_text: str,
    vocab: Dict[str, List[str]],
    threshold: float = 0.75,
    prefer_negative_if_no: bool = False,
    negative_threshold: Optional[float] = None,
) -> Optional[str]:
    """Retorna la clave cuyo conjunto de sinónimos tiene el mejor match fuzzy.

    Si *prefer_negative_if_no* es ``True`` y el input contiene el token ``no``,
    se evalúan primero los sinónimos negativos (que empiezan con ``"no "``).
    """
    input_norm = normalize_text(input_text)
    tokens = set(input_norm.split())

    def _find_best(candidates: Dict[str, List[str]], min_thr: float) -> Optional[str]:
        best_key, best_ratio = None, 0.0
        for key, terms in candidates.items():
            for term in terms:
                ratio = similarity_ratio(input_norm, normalize_text(term))
                if ratio > best_ratio:
                    best_ratio, best_key = ratio, key
        return best_key if best_key is not None and best_ratio >= min_thr else None

    # Priorizar variantes negativas cuando el texto contiene "no".
    if prefer_negative_if_no and "no" in tokens:
        neg_vocab: Dict[str, List[str]] = {}
        for key, terms in vocab.items():
            neg_terms = [
                t for t in terms
                if normalize_text(t).startswith("no ") or " no " in normalize_text(t)
            ]
            if neg_terms:
                neg_vocab[key] = neg_terms
        if neg_vocab:
            neg_thr = threshold if negative_threshold is None else negative_threshold
            neg_match = _find_best(neg_vocab, neg_thr)
            if neg_match is not None:
                return neg_match

    return _find_best(vocab, threshold)


# ─── Helpers de clasificación ──────────────────────────────────────────────────

def is_generic_indicator(value: Optional[str]) -> bool:
    """``True`` si *value* es vacío o no representa explícitamente IMACEC/PIB."""
    if not value:
        return True

    norm = normalize_text(value)

    # Únicos indicadores explícitos soportados por el flujo actual.
    if fuzzy_match(norm, ["imacec"], threshold=0.72):
        return False
    if fuzzy_match(norm, ["pib", "producto interno bruto"], threshold=0.72):
        return False

    # Cualquier otro valor se considera genérico para forzar inferencia.
    return True
