FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x docker/entrypoint.sh
# entrypoint.sh forwards all arguments to `python -m src.training.train_cli`
# (defaults: --model $MODEL --seed $SEED --config $CONFIG).
ENTRYPOINT ["docker/entrypoint.sh"]
