FROM python:3.11-slim
WORKDIR /app
COPY training_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY training_service/ /app/
CMD ["uvicorn", "train:app", "--host", "0.0.0.0", "--port", "8000"]
