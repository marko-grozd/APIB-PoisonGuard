FROM python:3.11-slim
WORKDIR /app
COPY serving_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY serving_api/ /app/
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
