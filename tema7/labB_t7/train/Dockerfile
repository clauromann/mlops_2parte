FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY train/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY utils/ ./utils/
COPY train/train.py ./train/train.py

CMD ["python", "train/train.py"]
