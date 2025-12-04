"""
app.py
------
Lógica de la aplicación Streamlit (frontend del chatbot).

- Usa Settings (config.py) para configurar nombre del bot y parámetros.
- Recibe funciones de orquestación (stream_fn / invoke_fn) desde main.py.
- Maneja la historia de conversación en st.session_state.
"""
import uuid
from typing import Callable, List, Dict, Optional, Iterable
import datetime
import time
import os

import streamlit as st

from config import Settings
from pathlib import Path

# Nuevo: reutilizar el logger unificado del orquestador para escribir en el mismo log
try:
    import orchestrator as _orch
except Exception:
    _orch = None  # type: ignore

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
        welcome_text = (settings.welcome_message or "").strip()
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
            st.session_state.orch = st.session_state.get("orch") or _orch.create_orchestrator_with_langchain(model=model_sel, temperature=float(temp_sel))
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
        model_sel = st.text_input("Modelo", value=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
        temp_sel = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        if st.button("Nueva sesión", icon=":material/refresh:"):
            _clear_conversation()
            st.experimental_rerun()

    # Título y botón de restart
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.title(settings.bot_name, anchor=False)
    with col_btn:
        st.button("Restart", icon=":material/refresh:", on_click=_clear_conversation)

    # Pequeña descripción
    st.caption("Chatbot que responde a consultas del PIB y el IMACEC")

    # Mostrar historial de mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrada del usuario
    user_message = st.chat_input("Escribe tu pregunta...")
    if not user_message:
        return

    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_message)

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
    history: List[Dict[str, str]] = list(st.session_state.messages)

    # Llamar al orquestador en modo streaming y renderizar chunk a chunk
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            markers_csv: List[Dict[str, str]] = []
            markers_chart: List[Dict[str, str]] = []
            collecting_csv = False
            collecting_chart = False
            buffer_csv: List[str] = []
            buffer_chart: List[str] = []
            text_accum = ""
            placeholder = st.empty()
            # parámetros para animar el render incremental aunque el chunk sea grande
            _chunk_sleep = 0.03
            _chunk_size = 120

            def handle_chunk(chunk: str) -> None:
                nonlocal collecting_csv, collecting_chart, buffer_csv, buffer_chart, text_accum
                text = str(chunk)
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
                    if collecting_csv:
                        buffer_csv.append(ls)
                        continue
                    if collecting_chart:
                        buffer_chart.append(ls)
                        continue
                    out_lines.append(line)
                filtered = "".join(out_lines)
                if filtered:
                    # dividir en fragmentos para mostrar progresivo
                    for i in range(0, len(filtered), _chunk_size):
                        text_accum += filtered[i : i + _chunk_size]
                        placeholder.markdown(text_accum)
                        if len(filtered) > _chunk_size:
                            time.sleep(_chunk_sleep)

            raw_chunks = stream_fn(user_message, history=history, session_id=st.session_state.session_id)
            for _chunk in raw_chunks:
                handle_chunk(_chunk)
            response_text = text_accum

    # Almacenar markers en session_state (sin limpiar)
    st.session_state.csv_markers = markers_csv
    st.session_state.chart_markers = markers_chart

    # Renderizar descargas CSV
    if st.session_state.csv_markers:
        with st.chat_message("assistant"):
            st.markdown("### Descargas de datos")
            for i, b in enumerate(st.session_state.csv_markers, start=1):
                path = b.get("path")
                if not path:
                    continue
                filename = b.get("filename") or f"datos_{i}.csv"
                label = b.get("label") or "Descargar CSV"
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
                    key=f"download-csv-{i}-{filename}",
                )

    # Renderizar gráficos
    if st.session_state.chart_markers:
        with st.chat_message("assistant"):
            st.markdown("### Gráficos de la serie")
            for i, b in enumerate(st.session_state.chart_markers, start=1):
                path = b.get("data_path")
                if not path:
                    continue
                title = b.get("title", "Gráfico")
                chart_type = b.get("type", "line")
                domain = b.get("domain", "")
                try:
                    import pandas as _pd  # type: ignore
                    df = _pd.read_csv(path)
                    # Usar EXACTAMENTE las filas del CSV emitido (sin recortes adicionales)
                    # Derivar etiqueta de período sin añadir intervalos extra.
                    if "date" in df.columns:
                        try:
                            df["_dt"] = _pd.to_datetime(df["date"], errors="coerce")
                            # Detectar si son trimestres (meses 3,6,9,12) para etiqueta 'Tn/YYYY'
                            months_set = set(df["_dt"].dropna().dt.month.unique().tolist())
                            is_quarterly = months_set.issubset({3,6,9,12}) and len(months_set) <= 4
                            if is_quarterly:
                                q_map = {3:"T1",6:"T2",9:"T3",12:"T4"}
                                df["period"] = df["_dt"].apply(lambda d: f"{q_map.get(d.month,'?')}/{d.year}" if not _pd.isna(d) else None)
                            else:
                                df["period"] = df["_dt"].dt.strftime("%m/%Y")
                        except Exception:
                            pass
                    # Renombrar 'value' a 'indice' para leyenda
                    if "value" in df.columns:
                        df.rename(columns={"value": "indice"}, inplace=True)
                    # Escalar variación anual a porcentaje (%) si viene en fracción
                    if "yoy_pct" in df.columns:
                        try:
                            s = _pd.to_numeric(df["yoy_pct"], errors="coerce")
                            if s.notna().any():
                                max_abs = float(s.abs().max())
                                if max_abs < 1.0:  # probablemente fracción -> convertir a %
                                    s = s * 100.0
                                df["yoy_pct_%"] = s
                        except Exception:
                            pass
                except Exception:
                    df = None
                # Descripción previa
                st.markdown(f"**{title}**")
                # Descripción sin paréntesis
                st.caption(
                    "Descripción: variación anual en porcentaje del año consultado, sin índice ni año previo."
                    + (f" Dominio: {domain}." if domain else "")
                )
                if df is not None and chart_type == "line":
                    try:
                        # Selección de columnas (solo variación anual actual)
                        # Mostrar una sola serie en el gráfico para evitar leyendas duplicadas
                        cols_pref = ["yoy_pct_%", "yoy_pct"]
                        cols_plot_all = [c for c in cols_pref if c in df.columns]
                        cols_plot = cols_plot_all[:1] if cols_plot_all else []
                        if cols_plot:
                            # Usar 'period' (MM/YYYY) si está disponible; si no, 'date'
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

    # Guardar respuesta en historial (chat principal sin markers)
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": response_text})

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
                    hist2: List[Dict[str, str]] = list(st.session_state.messages)
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
                    st.experimental_rerun()
