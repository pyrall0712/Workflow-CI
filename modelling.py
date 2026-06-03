import os

def train_model():
    # Membaca token dari environment variable server GitHub Actions
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    
    if dagshub_token:
        # Jika berjalan di GitHub Actions, gunakan token untuk otentikasi non-interaktif
        os.environ["DAGSHUB_CLIENT_TOKEN"] = dagshub_token
    
    # Inisialisasi otomatis DagsHub
    dagshub.init(repo_owner='pyrall0712', repo_name='Eksperimen_SML_MuhamadRizal', mlflow=True)