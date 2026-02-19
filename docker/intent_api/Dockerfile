FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY intent_api ./intent_api

EXPOSE 8000

CMD ["uvicorn", "intent_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
