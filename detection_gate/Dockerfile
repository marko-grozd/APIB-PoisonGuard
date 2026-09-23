FROM python:3.11-slim
WORKDIR /app
COPY detection_gate/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY detection_gate/ /app/
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
