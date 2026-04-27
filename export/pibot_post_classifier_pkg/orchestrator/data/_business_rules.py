"""Compatibility shim. La lógica real vive en rules.post_classifier."""
from __future__ import annotations

from rules.post_classifier import (  # noqa: F401
    ResolvedEntities,
    apply_business_rules,
    _is_empty_cls,
    _SHARE_INTENT_RE,
    _YOY_KEYWORDS_RE,
    _VAR_MENSUAL_RE,
    _VAR_TRIMESTRAL_RE,
    _rule_detect_share_intent,
    _rule_natural_freq_variation_is_prev_period,
    _rule_seasonality_sa_implies_prev_period,
    _rule_contribution_investment_force_general,
    _rule_contribution_demanda_interna,
    _rule_imacec_force_monthly,
    _rule_imacec_default_activity,
    _rule_assign_price,
    _rule_pib_hist_flag,
    _rule_redirect_pib_monthly_to_quarterly,
)

__all__ = [
    "ResolvedEntities",
    "apply_business_rules",
    "_is_empty_cls",
    "_SHARE_INTENT_RE",
    "_YOY_KEYWORDS_RE",
    "_VAR_MENSUAL_RE",
    "_VAR_TRIMESTRAL_RE",
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
