FROM python:3.11-slim
WORKDIR /app
COPY attack_simulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY attack_simulator/ /app/
CMD ["python", "simulate.py"]
