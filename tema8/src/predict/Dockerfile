FROM python:3.11-slim

WORKDIR /app

COPY src/predict/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/predict/handler.py ./handler.py
COPY src/predict/model/ ./model/

CMD ["python", "handler.py"]
