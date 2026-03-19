from pathlib import Path

from orchestrator.graph.send_response import make_response_node


def test_response_node_generates_run_detail_log_for_pib_latest_quarter(monkeypatch):
    question = "cual es el valor del pib del ultimo trimestre"
    session_id = "test-session-run-detail"

    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_name = "run_detail_test.log"
    log_path = logs_dir / log_name
    if log_path.exists():
        log_path.unlink()

    monkeypatch.setenv("GRAPH_DETAIL_LOG_ENABLED", "1")
    monkeypatch.setenv("GRAPH_DETAIL_LOG_FILE", log_name)
    monkeypatch.setenv("RESPONSE_ACCEPTANCE_MIN_LEN", "20")

    node = make_response_node(rag_llm_adapter=None, fallback_llm_adapter=None)

    state = {
        "question": question,
        "session_id": session_id,
        "context": {"session_id": session_id},
        "route_decision": "data",
        "data_store_lookup": {
            "source_url": "https://si3.bcentral.cl/Siete/ES/Siete/Cuadro/CAP_CCNN/MN_CCNN76/CCNN2018_IMACEC_01_A",
            "frequency": "T",
            "series_count": 1,
        },
        "response_payload": {
            "mode": "prebuilt",
            "route_decision": "data",
            "text": (
                "En el 4to trimestre de 2025, el PIB registró una variación anual de **2,1%**. "
                "Como recomendación final: puedes revisar el detalle por actividad económica.\n\n"
                "**Fuente:** 🔗 [Base de Datos Estadísticos (BDE)]"
                "(https://si3.bcentral.cl/Siete/ES/Siete/Cuadro/CAP_CCNN/MN_CCNN76/CCNN2018_IMACEC_01_A)"
                " del Banco Central de Chile."
            ),
        },
    }

    result = node(state)

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")

    assert f"Session: {session_id}" in content
    assert f"Pregunta: {question}" in content
    assert "Tipo de clasificacion: data" in content
    assert "Respuesta final:" in content
    assert "Criterios de aceptacion:" in content
    assert "Criterio final:" in content
    assert result.get("response_detail_log_path", "").endswith(log_name)


def test_response_node_uses_default_run_detail_log_name(monkeypatch):
    question = "cual es el valor del pib del ultimo trimestre"
    session_id = "test-session-default-run-detail"

    repo_root = Path(__file__).resolve().parents[1]
    default_log_path = repo_root / "logs" / "run_detail.log"
    default_log_path.parent.mkdir(parents=True, exist_ok=True)
    before_size = default_log_path.stat().st_size if default_log_path.exists() else 0

    monkeypatch.setenv("GRAPH_DETAIL_LOG_ENABLED", "1")
    monkeypatch.delenv("GRAPH_DETAIL_LOG_FILE", raising=False)
    monkeypatch.setenv("RESPONSE_ACCEPTANCE_MIN_LEN", "20")

    node = make_response_node(rag_llm_adapter=None, fallback_llm_adapter=None)

    result = node(
        {
            "question": question,
            "session_id": session_id,
            "context": {"session_id": session_id},
            "route_decision": "data",
            "data_store_lookup": {"source_url": "https://si3.bcentral.cl"},
            "response_payload": {
                "mode": "prebuilt",
                "route_decision": "data",
                "text": "En el 4to trimestre de 2025, el PIB registró una variación anual de **2,1%**.",
            },
        }
    )

    assert result.get("response_detail_log_path", "").endswith("run_detail.log")
    assert default_log_path.exists()
    assert default_log_path.stat().st_size > before_size
