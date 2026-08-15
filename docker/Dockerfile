FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "-m", "src.training.train_cli"]

