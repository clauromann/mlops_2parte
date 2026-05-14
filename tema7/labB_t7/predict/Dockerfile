FROM python:3.11-slim

WORKDIR /app

COPY predict/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY predict/ ./predict/

CMD ["python", "predict/handler.py"]
