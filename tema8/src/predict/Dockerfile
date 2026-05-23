FROM python:3.11-slim

WORKDIR /app

COPY src/predict/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ARG WANDB_API_KEY
ARG WANDB_ENTITY

ENV WANDB_API_KEY=$WANDB_API_KEY
ENV WANDB_ENTITY=$WANDB_ENTITY

COPY src/predict/config.json ./config.json
COPY src/predict/download_model.py ./download_model.py
RUN python download_model.py

COPY src/predict/handler.py ./handler.py

CMD ["python", "handler.py"]
