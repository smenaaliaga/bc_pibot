from __future__ import annotations

import orchestrator.data.response as response_module


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content


def test_build_messages_uses_template_rewrite_mode(monkeypatch):
    monkeypatch.setattr(response_module, "SystemMessage", _DummyMessage)
    monkeypatch.setattr(response_module, "HumanMessage", _DummyMessage)

    template = (
        "La participacion del PIB por componentes del gasto requiere revisar las series "
        "publicadas. Las series las puedes revisar en el siguiente cuadro segun los datos "
        "de la BDE publicados por el Banco Central de Chile: "
        "https://si3.bcentral.cl/Siete/ES/Siete/Cuadro/CAP_CCNN/MN_CCNN76/CCNN_EP18_03_ratio"
    )

    messages = response_module._build_messages(
        {
            "question": "Cuanto pesa el consumo en el PIB?",
            "response_template": template,
            "response_feature": "pib_share_gasto_unavailable",
        }
    )

    assert len(messages) == 2
    system_content = str(messages[0].content)
    user_content = str(messages[1].content)

    assert "plantilla" in system_content.lower()
    assert "similar" in system_content.lower()
    assert "template base" in system_content.lower()
    assert template in user_content
    assert "URL obligatoria" in user_content
    assert "CCNN_EP18_03_ratio" in user_content


def test_extract_first_url_from_template_text():
    text = (
        "Puedes revisar la informacion aqui: "
        "https://si3.bcentral.cl/Siete/ES/Siete/Cuadro/CAP_CCNN/MN_CCNN76/CCNN_EP18_03_ratio"
    )
    url = response_module._extract_first_url(text)
    assert url == "https://si3.bcentral.cl/Siete/ES/Siete/Cuadro/CAP_CCNN/MN_CCNN76/CCNN_EP18_03_ratio"


def test_build_system_prompt_original_prioritizes_yoy():
    prompt = response_module._build_system_prompt(calc_mode="original")

    assert "Regla de campo prioritario (calc_mode=original)" in prompt
    assert "campo principal a reportar es 'yoy_pct'" in prompt
    assert "NO menciones el 'value'" in prompt


def test_build_messages_latest_original_uses_yoy_hint(monkeypatch):
    monkeypatch.setattr(response_module, "SystemMessage", _DummyMessage)
    monkeypatch.setattr(response_module, "HumanMessage", _DummyMessage)

    messages = response_module._build_messages(
        {
            "question": "Cual es el ultimo PIB?",
            "classification": {
                "calc_mode_cls": "original",
                "req_form_cls": "latest",
                "indicator_ent": "pib",
                "price": "co",
            },
            "series": "SERIE.X",
            "series_title": "PIB a precios corrientes",
            "observations": {
                "SERIE.X": {
                    "meta": {
                        "descripEsp": "PIB a precios corrientes",
                        "original_frequency": "Q",
                    },
                    "observations": [
                        {"date": "2025-03-31", "value": 100.0, "yoy_pct": 3.4, "pct": 0.8},
                    ],
                }
            },
        }
    )

    assert len(messages) == 2
    system_content = str(messages[0].content)
    user_content = str(messages[1].content)

    assert "campo principal a reportar es 'yoy_pct'" in system_content
    assert "NO menciones el 'value'" in system_content
    assert "fecha=2025-03-31, yoy_pct=3.4" in user_content
    assert "El nivel del índice (value=" not in user_content
