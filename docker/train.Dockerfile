FROM python:3.10-slim

WORKDIR /app

# Log explícito de build (debug)
RUN python --version

RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    google-cloud-storage \
    pandas 

COPY training/train.py /app/train.py

# Força stdout não-bufferizado
CMD ["python", "-u", "/app/train.py"]
