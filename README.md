# PIBot BCCh – Chat IMACEC/PIB
**Build:** 2025-10-27 18:32:31

## Instalación
```bash
python -m venv .venv
# Windows: 
.venv\Scripts\activate
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
