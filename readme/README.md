# PIBot – Chat IMACEC/PIB (Function Calling + BCCh)
**Build:** 2025-10-27 18:32:31

## Novedades clave
- **Valores "tal como la API"** cuando no pides variación (ej.: *"resumen del IMACEC 2025"*).
- **YoY solo cuando lo pides** (mismo período del año anterior) con casteo numérico seguro.
- **Referencia del cuadro** del BCCh incluida al final si la serie tiene `source_url`.


## Instalación
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Completa OPENAI_API_KEY, BCCH_USER y BCCH_PASS
```

## Ejecución
```bash
streamlit run main.py
```
