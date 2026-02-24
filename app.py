"""
app.py
------
Lógica de la aplicación Streamlit (frontend del chatbot).

- Usa Settings (config.py) para configurar nombre del bot y parámetros.
- Recibe funciones de orquestación (stream_fn / invoke_fn) desde main.py.
- Maneja la historia de conversación en st.session_state.
"""
import logging
import uuid
import json
from typing import Callable, List, Dict, Optional, Iterable
import datetime
import time
import os
import re

import streamlit as st

from config import Settings
from orchestrator.memory.memory_adapter import MemoryAdapter
from pathlib import Path

# Nuevo: reutilizar el logger unificado del orquestador para escribir en el mismo log
try:
    import orchestrator as _orch
except Exception:
    _orch = None  # type: ignore

# Resolver logger para la UI: preferir el logger del orquestador; si no tiene
# handlers, caer al root y, en última instancia, configurar uno básico.
def _resolve_ui_logger() -> logging.Logger:
    candidates = []
    if _orch and hasattr(_orch, "logger"):
        try:
            candidates.append(_orch.logger)  # type: ignore[attr-defined]
        except Exception:
            pass
    candidates.append(logging.getLogger())  # root
    candidates.append(logging.getLogger(__name__))
    for lg in candidates:
        if lg and lg.handlers:
            return lg
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger()

_ui_logger = _resolve_ui_logger()

# Tipos para las funciones de orquestación
StreamFn = Callable[[str, Optional[List[Dict[str, str]]], Optional[str]], Iterable[str]]
InvokeFn = Callable[[str, Optional[List[Dict[str, str]]]], str]


def _init_session_state(settings: Settings) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # type: ignore[assignment]

    if "prev_question_timestamp" not in st.session_state:
        st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

    if "settings" not in st.session_state:
        st.session_state.settings = settings  # type: ignore[assignment]
    else:
        # Asegura que la configuración actualizada se mantenga disponible
        st.session_state.settings = settings  # type: ignore[assignment]

    if "welcome_emitted" not in st.session_state:
        st.session_state.welcome_emitted = False

    # Generar session_id solo una vez por sesión de navegador
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"st-{uuid.uuid4().hex}"

    if not st.session_state.welcome_emitted:
        welcome_text = (getattr(settings, "welcome_message", "") or "").strip()
        if not welcome_text:
            welcome_text = (
                f"Hola, soy {settings.bot_name}, asistente económico del Banco Central de Chile. "
                "¿En qué puedo ayudarte hoy?"
            )
        st.session_state.messages.append({"role": "assistant", "content": welcome_text})
        st.session_state.welcome_emitted = True


def _clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)
    st.session_state.welcome_emitted = False
    if "orch" in st.session_state:
        st.session_state.pop("orch")
    # Forzar nuevo session_id para una conversación limpia
    st.session_state.session_id = f"st-{uuid.uuid4().hex}"


def run_app(
    settings: Settings,
    stream_fn: Optional[StreamFn] = None,
    invoke_fn: Optional[InvokeFn] = None,  # noqa: ARG001 (por ahora no lo usamos, pero queda disponible)
) -> None:
    """
    Punto de entrada de la app Streamlit.

    Debe ser llamado desde main.py, pasando:
    - settings: configuración cargada desde config.py
    - stream_fn: función de LangChain para streaming
    - invoke_fn: función de LangChain para invocación sin streaming
    """

    # Debe ser lo primero que se ejecuta en Streamlit
    # Usar logo por defecto si existe
    from pathlib import Path as _Path
    def _resolve_page_icon() -> str:
        candidates = [
            _Path("assets/logo.png"),
            _Path("assets/icon.png"),
            _Path("logo.png"),
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        return "✨"

    st.set_page_config(page_title=settings.bot_name, page_icon=_resolve_page_icon())
    _init_session_state(settings)

    # Crear orquestador LangChain si no se pasó stream_fn externo
    if stream_fn is None:
        if not _orch:
            st.error("No se pudo cargar el módulo orchestrator.")
            return
        try:
            # Usar valores por defecto desde entorno para evitar depender del orden del sidebar
            _model_default = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            try:
                _temp_default = float(os.getenv("OPENAI_TEMPERATURE", "0") or 0.0)
            except Exception:
                _temp_default = 0.0
            st.session_state.orch = st.session_state.get("orch") or _orch.create_orchestrator_with_langchain(model=_model_default, temperature=_temp_default)
            try:
                import logging as _logging
                if hasattr(_orch, "logger"):
                    _orch.logger = _logging.LoggerAdapter(_orch.logger, extra={"session_id": st.session_state.session_id})  # type: ignore
            except Exception:
                pass
            stream_fn = lambda q, history=None, session_id=None: st.session_state.orch.stream(q, history=history, session_id=session_id)  # type: ignore[assignment]
        except Exception as e:
            st.error(f"No se pudo inicializar el orquestador: {e}")
            return

    # Encabezado: mostrar logo si está disponible
    _logo_path = _Path("assets/logo.png")
    if _logo_path.is_file():
        st.image(str(_logo_path), width=120)
    else:
        st.markdown(
            "<div style='font-size:3rem; line-height:1;'>❉</div>",
            unsafe_allow_html=True,
        )

    # Barra lateral: ajustes de modelo/temperatura y acciones de sesión
    with st.sidebar:
        st.subheader("Debug")
        st.write(f"Session ID: `{st.session_state.session_id}`")
        
        st.subheader("Modelo generativo")
        model_sel = st.text_input("Modelo", value=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
        temp_sel = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=float(os.getenv("OPENAI_TEMPERATURE", "0") or 0.0), step=0.1)
        
        st.subheader("Modelo predictror")
        # Mostrar solo el nombre de la carpeta final del BERT base/tokenizer, sin permitir edición
        bert_model_full = os.getenv("BERT_MODEL_NAME", "")
        bert_model_name = os.path.basename(bert_model_full.rstrip("/\\")) if bert_model_full else ""
        st.text_input("Modelo BERT", value=bert_model_name)
        # Mostrar solo el nombre de la carpeta final, pero mantener el path completo
        joint_bert_model_dir_full = os.getenv("JOINT_BERT_MODEL_DIR", "")
        joint_bert_model_dir_name = os.path.basename(joint_bert_model_dir_full.rstrip("/\\")) if joint_bert_model_dir_full else ""
        joint_bert_model_dir_name_new = st.text_input("Modelo Joint BERT", value=joint_bert_model_dir_name)
        # Reconstruir el path completo si el usuario lo cambia
        if joint_bert_model_dir_name_new != joint_bert_model_dir_name and joint_bert_model_dir_name_new:
            # Mantener el directorio padre original si existe, si no, usar el valor nuevo tal cual
            parent_dir = os.path.dirname(joint_bert_model_dir_full) if joint_bert_model_dir_full else "models/pibot_series_interpreter"
            joint_bert_model_dir = os.path.join(parent_dir, joint_bert_model_dir_name_new)
        else:
            joint_bert_model_dir = joint_bert_model_dir_full
        
        # Aplicar cambios de modelo al entorno para que el predictor los tome
        _env_changed = False
        if model_sel and os.getenv("OPENAI_MODEL") != model_sel:
            os.environ["OPENAI_MODEL"] = model_sel
            _env_changed = True
        # Persistir temperatura elegida
        if os.getenv("OPENAI_TEMPERATURE") != str(temp_sel):
            os.environ["OPENAI_TEMPERATURE"] = str(temp_sel)
            _env_changed = True
        # Tokenizer/base model para JointBERT (solo lectura desde UI)
        # Directorio del modelo entrenado de JointBERT
        if joint_bert_model_dir and os.getenv("JOINT_BERT_MODEL_DIR") != joint_bert_model_dir:
            os.environ["JOINT_BERT_MODEL_DIR"] = joint_bert_model_dir
            _env_changed = True

        # --- Panel dinámico: Memoria y clasificación ---------------------
        def _get_mem_adapter() -> MemoryAdapter:
            _ma = st.session_state.get("_mem_adapter")
            if not _ma:
                _ma = MemoryAdapter(pg_dsn=os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot"))
                st.session_state._mem_adapter = _ma
            return _ma  # type: ignore

        def _render_memory_debug(dst_container):
            try:
                _ma = _get_mem_adapter()
                _sid = st.session_state.session_id
                # Header
                dst_container.subheader("Clasificación y memoria")
                # Backend
                try:
                    backend = _ma.get_backend_status() or {}
                except Exception:
                    backend = {}
                using_pg = bool(backend.get("using_pg"))
                dsn = str(backend.get("dsn") or "")
                # dst_container.caption(f"Backend: {'Postgres' if using_pg else 'Local'}{(' · ' + dsn) if dsn else ''}")
                # Facts
                try:
                    facts = _ma.get_facts(_sid) or {}
                except Exception:
                    facts = {}
                
                # Deserializar facts si vienen como strings JSON
                facts_display = {}
                for k, v in facts.items():
                    if isinstance(v, str):
                        try:
                            # Intentar parsear como JSON
                            facts_display[k] = json.loads(v)
                        except Exception:
                            # Si no es JSON válido, mostrar como string
                            facts_display[k] = v
                    else:
                        facts_display[k] = v
                
                if facts_display:
                    # dst_container.markdown("**Clasificación (memoria)**")
                    dst_container.json(facts_display)
                else:
                    dst_container.caption("Sin facts de clasificación en memoria")
            except Exception as _e_mem:
                dst_container.caption(f"Memoria no disponible: {str(_e_mem)[:200]}")

        # Placeholder del panel para poder refrescarlo dinámicamente
        _mem_debug_ph = st.empty()
        st.session_state._mem_debug_placeholder = _mem_debug_ph
        _render_memory_debug(_mem_debug_ph.container())

        if st.button("Nueva sesión", icon=":material/refresh:"):
            _clear_conversation()
            st.rerun()

    # Título y botón de restart
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.title(settings.bot_name, anchor=False)
    with col_btn:
        st.button("Restart", icon=":material/refresh:", on_click=_clear_conversation)

    # Helper: extraer followups embebidos en contenido de asistente
    import re as _re

    def _extract_followups_from_text(text: str) -> tuple[str, List[Dict[str, str]]]:
        cleaned_parts: List[str] = []
        found: List[Dict[str, str]] = []
        pattern = _re.compile(r"##FOLLOWUP_START(.*?)##FOLLOWUP_END", _re.DOTALL)
        last_idx = 0
        for m in pattern.finditer(text):
            cleaned_parts.append(text[last_idx : m.start()])
            block = m.group(1)
            d: Dict[str, str] = {}
            for ln in block.splitlines():
                ln = ln.strip()
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    d[k.strip()] = v.strip()
            if d:
                found.append(d)
            last_idx = m.end()
        cleaned_parts.append(text[last_idx:])
        return ("".join(cleaned_parts), found)

    # Pequeña descripción
    st.caption("Chatbot que responde a consultas del PIB y el IMACEC")

    # Helper: construir followups en UI a partir de facts si no vienen marcadores
    def _build_ui_followups_from_facts() -> List[Dict[str, str]]:
        suggestions: List[str] = []
        try:
            _ma = st.session_state.get("_mem_adapter")
            if _ma:
                facts = _ma.get_facts(st.session_state.session_id) or {}
                indicator = facts.get("indicator")
                component = facts.get("component")
                seasonality = facts.get("seasonality")
                period = facts.get("period")

                if not indicator:
                    suggestions.extend([
                        "Cuanto aceleró la economía el último mes",
                        "Explícame que es el PIB",
                    ])
                else:
                    ind_lower = str(indicator).lower()
                    # Estacionalidad
                    if seasonality and "sa" in str(seasonality).lower():
                        suggestions.append(f"Cuanto creció el {indicator.upper()}")
                    else:
                        suggestions.append(f"Cuanto creció el {indicator.upper()} desestacionalizado")
                    # Específico IMACEC
                    if "imacec" in ind_lower:
                        comp_lower = str(component or "").lower()
                        if not comp_lower or comp_lower == "total":
                            suggestions.append("Cuanto creció el IMACEC minero")
                        elif "minero" in comp_lower:
                            suggestions.append("Cuanto varió el IMACEC no minero")
                        else:
                            suggestions.append("Cuanto creció el IMACEC")
                    # Específico PIB
                    if "pib" in ind_lower and not component:
                        suggestions.append("¿Cuál es la variación del PIB por sectores?")
                    # Metodología / general
                    suggestions.append(f"¿Qué mide el {indicator.upper()}?")
                    if period:
                        suggestions.append(f"¿Cómo ha evolucionado el {indicator.upper()} en los últimos años?")
            # Dedup y limitar
            seen = set()
            uniq = []
            for s in suggestions:
                key = re.sub(r"[^a-z0-9]+", "", s.lower())
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(s)
            return [{f"suggestion_{i+1}": s} for i, s in enumerate(uniq[:2])] # SOLO MUESTRA 2 PREGUNTAS
        except Exception:
            return []

    import re as _decor_re

    def _decorate_links(text: str) -> str:
        """Añade emoji de link a la referencia de la BDE de forma idempotente."""
        return _decor_re.sub(r"(?<!🔗 )Ver serie en la BDE", "🔗 Ver serie en la BDE", text)

    # Mostrar historial de mensajes (sin volver a renderizar botones de followup históricos)
    for idx_msg, msg in enumerate(st.session_state.messages):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant" and "##FOLLOWUP_START" in content:
            clean_text, parsed_followups = _extract_followups_from_text(content)
            with st.chat_message(role):
                if clean_text.strip():
                    st.markdown(_decorate_links(clean_text))
        else:
            with st.chat_message(role):
                st.markdown(_decorate_links(content))

    def _render_post_response_blocks(scope: str = "post") -> None:
        """Renderiza descargas, gráficos y preguntas sugeridas actuales."""
        import hashlib

        # Descargas CSV
        csv_markers_now = st.session_state.get("csv_markers") or []
        if csv_markers_now:
            with st.chat_message("assistant"):
                st.markdown("### 💾 Descargas de datos")
                for i, b in enumerate(csv_markers_now, start=1):
                    path = b.get("path")
                    if not path:
                        continue
                    filename = b.get("filename") or f"datos_{i}.csv"
                    label = b.get("label") or "💾 Descargar CSV"
                    mimetype = b.get("mimetype") or "text/csv"
                    try:
                        import pandas as _pd  # type: ignore
                        df = _pd.read_csv(path)
                        data_bytes = df.to_csv(index=False).encode("utf-8")
                    except Exception:
                        try:
                            data_bytes = Path(str(path)).read_bytes()
                        except Exception:
                            data_bytes = b""
                    st.download_button(
                        label,
                        data_bytes,
                        file_name=filename,
                        mime=mimetype,
                        key=f"download-{scope}-{i}-{hashlib.md5((filename + str(path)).encode()).hexdigest()[:8]}",
                    )

        # Gráficos
        chart_markers_now = st.session_state.get("chart_markers") or []
        if chart_markers_now:
            with st.chat_message("assistant"):
                st.markdown("### Gráficos de la serie")
                for i, b in enumerate(chart_markers_now, start=1):
                    path = b.get("data_path")
                    if not path:
                        continue
                    title = b.get("title", "Gráfico")
                    chart_type = b.get("type", "line")
                    domain = b.get("domain", "")
                    try:
                        import pandas as _pd  # type: ignore
                        df = _pd.read_csv(path)
                        if "date" in df.columns:
                            try:
                                df["_dt"] = _pd.to_datetime(df["date"], errors="coerce")
                                months_set = set(df["_dt"].dropna().dt.month.unique().tolist())
                                is_quarterly = months_set.issubset({3, 6, 9, 12}) and len(months_set) <= 4
                                if is_quarterly:
                                    q_map = {3: "T1", 6: "T2", 9: "T3", 12: "T4"}
                                    df["period"] = df["_dt"].apply(
                                        lambda d: f"{q_map.get(d.month, '?')}/{d.year}" if not _pd.isna(d) else None
                                    )
                                else:
                                    df["period"] = df["_dt"].dt.strftime("%m/%Y")
                            except Exception:
                                pass
                        if "value" in df.columns:
                            df.rename(columns={"value": "indice"}, inplace=True)
                        if "yoy_pct" in df.columns:
                            try:
                                s = _pd.to_numeric(df["yoy_pct"], errors="coerce")
                                if s.notna().any():
                                    max_abs = float(s.abs().max())
                                    if max_abs < 1.0:
                                        s = s * 100.0
                                    df["yoy_pct_%"] = s
                            except Exception:
                                pass
                    except Exception:
                        df = None
                    st.markdown(f"**{title}**")
                    st.caption(
                        "Descripción: variación anual en porcentaje del año consultado, sin índice ni año previo." +
                        (f" Dominio: {domain}." if domain else "")
                    )
                    if df is not None and chart_type == "line":
                        try:
                            cols_pref = ["yoy_pct_%", "yoy_pct"]
                            cols_plot_all = [c for c in cols_pref if c in df.columns]
                            cols_plot = cols_plot_all[:1] if cols_plot_all else []
                            if cols_plot:
                                idx_col = "period" if "period" in df.columns else "date"
                                st.line_chart(df.set_index(idx_col)[cols_plot])
                            else:
                                st.line_chart(df)
                        except Exception as _e_plot:
                            if _orch and hasattr(_orch, "logger"):
                                _orch.logger.error(f"[UI_CHART_ERROR] {_e_plot}")
                            st.caption("No se pudo renderizar el gráfico.")
                    else:
                        st.caption("Datos no disponibles para el gráfico.")

        # Preguntas sugeridas (usar fallback si no hay markers)
        followup_blocks_now = st.session_state.get("followup_markers") or _build_ui_followups_from_facts()
        if followup_blocks_now:
            with st.chat_message("assistant"):
                st.markdown("### 💡 Preguntas sugeridas")
                for block_idx, marker_dict in enumerate(followup_blocks_now):
                    suggestions = []
                    for key in sorted(marker_dict.keys()):
                        if key.startswith("suggestion_"):
                            suggestions.append(marker_dict[key])
                    if suggestions:
                        cols = st.columns(len(suggestions) if len(suggestions) <= 2 else 2)
                        for idx, suggestion in enumerate(suggestions[:2]):
                            col_idx = idx % 3
                            with cols[col_idx]:
                                import hashlib
                                button_key = f"followup_{scope}_{block_idx}_{idx}_{hashlib.md5(suggestion.encode()).hexdigest()[:8]}"
                                if st.button(
                                    suggestion,
                                    key=button_key,
                                    use_container_width=True,
                                    type="secondary",
                                ):
                                    st.session_state.pending_question = suggestion
                                    st.rerun()

    # Renderizar bloques con marcadores actuales (permite capturar clicks de followups)
    _render_post_response_blocks(scope="main")

    # Entrada del usuario SIEMPRE visible
    user_input_display = st.chat_input("Escribe tu pregunta...")

    # Resolver mensaje a procesar (prioriza pending_question)
    if st.session_state.get("pending_question"):
        user_message = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        user_message = user_input_display

    if not user_message:
        return

    # Nota: no limpiar followups aquí; se reemplazan cuando llega la nueva respuesta

    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_message)

    # Registrar turno de usuario en memoria
    try:
        _mem_adapter = st.session_state.get("_mem_adapter") or MemoryAdapter(pg_dsn=os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot"))
        st.session_state._mem_adapter = _mem_adapter
        _mem_adapter.on_user_turn(
            st.session_state.session_id,
            user_message,
            metadata={
                "source": "ui",
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                "temperature": os.getenv("OPENAI_TEMPERATURE", "0"),
            },
        )
        # Refrescar panel de memoria tras turno de usuario
        try:
            _ph = st.session_state.get("_mem_debug_placeholder")
            if _ph:
                _ph.empty()
                _render_memory_debug(_ph.container())
        except Exception:
            pass
    except Exception:
        pass

    # Rate limiting simple
    now = datetime.datetime.now()
    delta = now - st.session_state.prev_question_timestamp
    min_delta = datetime.timedelta(seconds=settings.min_time_between_requests)
    st.session_state.prev_question_timestamp = now

    if delta < min_delta:
        wait_for = (min_delta - delta).total_seconds()
        with st.spinner(f"Esperando {wait_for:.1f}s para respetar el rate limit..."):
            time.sleep(wait_for)

    # Historial previo (antes de agregar el mensaje actual)
    # Paridad con qa/qa.py: por defecto no enviar historial al clasificador/grafo.
    use_history = os.getenv("STREAMLIT_PASS_HISTORY_TO_GRAPH", "0").lower() in {"1", "true", "yes", "on"}
    history: List[Dict[str, str]] = list(st.session_state.messages) if use_history else []

    assistant_box = st.chat_message("assistant")
    status_placeholder = assistant_box.empty()
    status_placeholder.caption("Pensando...")
    markers_csv: List[Dict[str, str]] = []
    markers_chart: List[Dict[str, str]] = []
    markers_followup: List[Dict[str, str]] = []
    collecting_csv = False
    collecting_chart = False
    collecting_followup = False
    buffer_csv: List[str] = []
    buffer_chart: List[str] = []
    buffer_followup: List[str] = []
    text_accum = ""
    placeholder = assistant_box.empty()
    _debug_chunk_idx = 0

    def handle_chunk(chunk: str) -> None:
        nonlocal collecting_csv, collecting_chart, collecting_followup, buffer_csv, buffer_chart, buffer_followup, text_accum, _debug_chunk_idx
        text = str(chunk)
        _debug_chunk_idx += 1
        try:
            preview = text[:200].replace("\n", "\\n")
            if os.getenv("STREAM_CHUNK_LOGS", "0").lower() in {"1", "true", "yes", "on"}:
                _ui_logger.debug(
                    "[UI_STREAM_CHUNK] idx=%s len=%s preview=%s",
                    _debug_chunk_idx,
                    len(text),
                    text[:120].replace("\n", " "),
                )
                _ui_logger.debug(
                    "[UI_STREAM_CHUNK_RAW] idx=%s repr=%s",
                    _debug_chunk_idx,
                    preview,
                )
        except Exception:
            pass
        out_lines: List[str] = []
        for line in text.splitlines(keepends=True):
            ls = line.strip()
            if ls == "##CSV_DOWNLOAD_START":
                collecting_csv = True
                buffer_csv = []
                continue
            if ls == "##CSV_DOWNLOAD_END":
                collecting_csv = False
                d: Dict[str, str] = {}
                for ln in buffer_csv:
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        d[k.strip()] = v.strip()
                if d:
                    markers_csv.append(d)
                buffer_csv = []
                continue
            if ls == "##CHART_START":
                collecting_chart = True
                buffer_chart = []
                continue
            if ls == "##CHART_END":
                collecting_chart = False
                d2: Dict[str, str] = {}
                for ln in buffer_chart:
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        d2[k.strip()] = v.strip()
                if d2:
                    markers_chart.append(d2)
                buffer_chart = []
                continue
            if ls == "##FOLLOWUP_START":
                collecting_followup = True
                buffer_followup = []
                continue
            if ls == "##FOLLOWUP_END":
                collecting_followup = False
                d3: Dict[str, str] = {}
                for ln in buffer_followup:
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        d3[k.strip()] = v.strip()
                if d3:
                    markers_followup.append(d3)
                buffer_followup = []
                continue
            if collecting_csv:
                buffer_csv.append(ls)
                continue
            if collecting_chart:
                buffer_chart.append(ls)
                continue
            if collecting_followup:
                buffer_followup.append(ls)
                continue
            out_lines.append(line)
        filtered = "".join(out_lines)
        if filtered or not text_accum:
            text_accum += filtered
            placeholder.markdown(text_accum or "\u200B")

    raw_chunks = stream_fn(user_message, history=history, session_id=st.session_state.session_id)
    for _chunk in raw_chunks:
        handle_chunk(_chunk)
    response_text = _decorate_links(text_accum)

    # Fallback: extraer y limpiar marcadores de followup por si algún chunk los mostró como texto
    if "##FOLLOWUP_START" in response_text:
        import re as _re

        def _extract_followups_from_text(text: str) -> tuple[str, List[Dict[str, str]]]:
            cleaned_parts: List[str] = []
            found: List[Dict[str, str]] = []
            pattern = _re.compile(r"##FOLLOWUP_START(.*?)##FOLLOWUP_END", _re.DOTALL)
            last_idx = 0
            for m in pattern.finditer(text):
                cleaned_parts.append(text[last_idx : m.start()])
                block = m.group(1)
                d: Dict[str, str] = {}
                for ln in block.splitlines():
                    ln = ln.strip()
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        d[k.strip()] = v.strip()
                if d:
                    found.append(d)
                last_idx = m.end()
            cleaned_parts.append(text[last_idx:])
            return ("".join(cleaned_parts), found)

        response_text, parsed_followups = _extract_followups_from_text(response_text)
        if parsed_followups:
            markers_followup.extend(parsed_followups)
            # Re-render sin los marcadores para evitar que queden visibles
            placeholder.markdown(response_text or "\u200B")

    try:
        if os.getenv("STREAM_CHUNK_LOGS", "0").lower() in {"1", "true", "yes", "on"}:
            _ui_logger.debug(
                "[UI_STREAM_END] total_len=%s preview=%s",
                len(response_text),
                response_text[:200].replace("\n", " "),
            )
    except Exception:
        pass
    status_placeholder.empty()

    # Almacenar markers en session_state (sin limpiar)
    st.session_state.csv_markers = markers_csv
    st.session_state.chart_markers = markers_chart
    st.session_state.followup_markers = markers_followup

    # Registrar turno del asistente en memoria
    try:
        _mem_adapter = st.session_state.get("_mem_adapter") or MemoryAdapter(pg_dsn=os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot"))
        st.session_state._mem_adapter = _mem_adapter
        _mem_adapter.on_assistant_turn(
            st.session_state.session_id,
            response_text,
            metadata={
                "source": "ui",
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                "temperature": os.getenv("OPENAI_TEMPERATURE", "0"),
            },
        )
        # Refrescar panel de memoria tras turno del asistente
        try:
            _ph = st.session_state.get("_mem_debug_placeholder")
            if _ph:
                _ph.empty()
                _render_memory_debug(_ph.container())
        except Exception:
            pass
    except Exception:
        pass

    # Guardar respuesta en historial (chat principal sin markers)
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Forzar rerender para que los nuevos followups/descargas se muestren y sean clicables
    st.rerun()

    # (Botones de followup ahora se renderizan antes del input; se omite duplicado aquí)

    # Vector search: detectar coincidencias en el texto ya filtrado
    if "vector_matches" not in st.session_state:
        st.session_state.vector_matches = []
    if "vm_block_id" not in st.session_state:
        st.session_state.vm_block_id = None

    matches_now: List[Dict[str, object]] = []
    if response_text and "##VECTOR_MATCHES_START" in response_text:
        try:
            block = response_text.split("##VECTOR_MATCHES_START", 1)[1].split("##VECTOR_MATCHES_END", 1)[0]
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            import re as _re
            for ln in lines:
                m = _re.match(r"(\d+)\.\s+(.+?)\s+\(([A-Z0-9_.]+)\)$", ln)
                if m:
                    matches_now.append({"rank": int(m.group(1)), "title": m.group(2), "code": m.group(3)})
        except Exception:
            matches_now = []
        st.session_state.vector_matches = matches_now
        if matches_now:
            import time as _time
            st.session_state.vm_block_id = str(int(_time.time()))
    else:
        matches_now = st.session_state.vector_matches

    if matches_now:
        with st.chat_message("assistant"):
            st.markdown("### Series sugeridas (vector search)")
            labels_map = {m["code"]: f"{m['rank']}. {m['title']} ({m['code']})" for m in matches_now}
            for m in matches_now:
                code = m["code"]
                rank = m["rank"]
                title = m["title"]
                label = labels_map.get(code, code)
                if st.button(f"Usar serie {label}", key=f"vm_btn_{code}"):
                    try:
                        if _orch and hasattr(_orch, "logger"):
                            _orch.logger.info(
                                "[UI_VECTOR_SELECT_BTN] click | " f"rank={rank} | code={code} | title={title}"
                            )
                    except Exception:
                        pass
                    st.caption(f"Serie seleccionada: {code}")
                    cmd = f"usar serie {code}"
                    with st.chat_message("user"):
                        st.markdown(cmd)
                    use_history2 = os.getenv("STREAMLIT_PASS_HISTORY_TO_GRAPH", "0").lower() in {"1", "true", "yes", "on"}
                    hist2: List[Dict[str, str]] = list(st.session_state.messages) if use_history2 else []
                    with st.chat_message("assistant"):
                        with st.spinner("Procesando los datos solicitados..."):
                            response_chunks = stream_fn(cmd, history=hist2, session_id=st.session_state.session_id)
                            # Render streaming en UI secundaria
                            text_accum2 = ""
                            ph2 = st.empty()
                            for ch2 in response_chunks:
                                text_accum2 += str(ch2)
                                ph2.markdown(text_accum2)
                            response_text2 = text_accum2
                    st.session_state.messages.append({"role": "user", "content": cmd})
                    st.session_state.messages.append({"role": "assistant", "content": response_text2})
                    st.session_state.vector_matches = []
                    st.session_state.vm_block_id = None
                    st.rerun()
