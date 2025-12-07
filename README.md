# PIBot BCCh – Chat IMACEC/PIB

## Requisitos
- Python 3.12 (gestionado con `uv`)
- Variables de entorno necesarias (`OPENAI_API_KEY`, `BCCH_USER`, `BCCH_PASS`, etc.).

## Instalación con uv
```bash
# Instala Python 3.12 si no lo tienes
uv python install 3.12

# Sincroniza dependencias y crea el entorno .venv
uv sync

# (Opcional) activa el entorno
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # macOS/Linux
```

Configura el archivo `.env` (puedes copiar desde `.env.example` si existe) y completa las claves requeridas.

## Ejecución
```bash
# Desde la raíz del repo
uv run streamlit run main.py
```
