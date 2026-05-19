"""Tests para sub-actividades de Industria Manufacturera y Minería en PIB.

Antes del fix de mayo 2026 el normalizer sólo conocía actividades de nivel
superior (industria, mineria, comercio, ...) y consultas como
"PIB de Celulosa, papel e imprentas" devolvían activity=[]; esto
contradecía la realidad porque la serie SÍ existe en el data_store
("PIB por clase de actividad económica" ⇒ short_title "Celulosa, papel e
imprentas").

Estos tests garantizan que el normalizer pobla activity_ent para las
sub-actividades publicadas en el desglose de PIB del Banco Central.
"""

import pytest

from orchestrator.normalizer.normalizer import normalize_entities


@pytest.mark.parametrize(
    "user_activity, expected_token",
    [
        # Sub-actividades de Industria Manufacturera
        ("Celulosa, papel e imprentas", "celulosa_papel_imprentas"),
        ("celulosa", "celulosa_papel_imprentas"),
        ("papel", "celulosa_papel_imprentas"),
        ("imprentas", "celulosa_papel_imprentas"),
        ("Alimentos, bebidas y tabaco", "alimentos_bebidas_tabaco"),
        ("alimentos", "alimentos_bebidas_tabaco"),
        ("bebidas", "alimentos_bebidas_tabaco"),
        ("tabaco", "alimentos_bebidas_tabaco"),
        ("Textil, prendas de vestir, cuero y calzado", "textil_vestir_cuero_calzado"),
        ("textil", "textil_vestir_cuero_calzado"),
        ("calzado", "textil_vestir_cuero_calzado"),
        ("Maderas y muebles", "maderas_muebles"),
        ("muebles", "maderas_muebles"),
        ("madera", "maderas_muebles"),
        ("Química, petróleo, caucho y plástico", "quimica_petroleo_caucho_plastico"),
        ("quimica", "quimica_petroleo_caucho_plastico"),
        ("petroleo", "quimica_petroleo_caucho_plastico"),
        ("caucho", "quimica_petroleo_caucho_plastico"),
        ("Minerales no metálicos y metálica básica", "minerales_no_metalicos_metalica_basica"),
        ("Productos metálicos, maquinaria, equipos y otros", "productos_metalicos_maquinaria"),
        ("maquinaria", "productos_metalicos_maquinaria"),
        # Sub-actividades de Minería
        ("Minería del cobre", "mineria_cobre"),
        ("cobre", "mineria_cobre"),
        ("Otras actividades mineras", "otras_actividades_mineras"),
    ],
)
def test_pib_subactivity_normalizes(user_activity: str, expected_token: str) -> None:
    """Verifica que las sub-actividades publicadas en el desglose de PIB se
    normalizan a un token canónico (no vacío).

    Esto evita que `_build_prevalidated_missing_specific_activity_instruction`
    se dispare cuando la actividad SÍ está disponible en el data_store.
    """
    entities = {
        "indicator": ["pib"],
        "activity": [user_activity],
        "period": ["primer trimestre 2026"],
    }

    normalized = normalize_entities(entities, calc_mode="original", req_form="latest")

    assert normalized["indicator"] == ["pib"]
    assert expected_token in normalized["activity"], (
        f"Sub-actividad '{user_activity}' debería normalizar a '{expected_token}' "
        f"pero se obtuvo {normalized['activity']}"
    )


def test_celulosa_query_produces_non_empty_activity() -> None:
    """Regresión exacta del bug reportado el 19 mayo 2026: la consulta
    'Cual es el valor del pib de Celulosa, papel e imprentas' debe poblar
    activity_ent (no devolver una lista vacía).
    """
    entities = {
        "indicator": ["pib"],
        "activity": ["Celulosa, papel e imprentas"],
        "period": ["1er trimestre 2026"],
    }
    normalized = normalize_entities(entities, calc_mode="original", req_form="latest")
    assert normalized["activity"], (
        "El bug original era que activity quedara vacío al pedir 'Celulosa, papel e imprentas'."
    )
    assert "celulosa_papel_imprentas" in normalized["activity"]
