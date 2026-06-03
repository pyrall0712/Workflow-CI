FROM python:3.12.7-slim

WORKDIR /app

# Memasang dependensi sistem yang diperlukan
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Menyalin berkas kebutuhan proyek
COPY conda.yaml /app/conda.yaml
COPY MLproject /app/MLproject
COPY modelling.py /app/modelling.py

# Memasang library utama
RUN pip install --no-cache-dir mlflow pandas scikit-learn dagshub

# Menjalankan perintah bawaan saat container aktif
CMD ["python", "modelling.py"]