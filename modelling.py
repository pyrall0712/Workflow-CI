import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import dagshub

def train_model():
    # 1. Otentikasi otomatis menggunakan token lingkungan GitHub Actions
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token:
        os.environ["DAGSHUB_CLIENT_TOKEN"] = dagshub_token
    
    # Inisialisasi koneksi MLflow ke DagsHub
    dagshub.init(repo_owner='pyrall0712', repo_name='Eksperimen_SML_MuhamadRizal', mlflow=True)

    # 2. Mengambil data hasil preprocessing dari repositori utama Anda
    url_train = "https://raw.githubusercontent.com/pyrall0712/Eksperimen_SML_MuhamadRizal/main/preprocessing/namadataset_preprocessing/iris_train_processed.csv"
    url_test = "https://raw.githubusercontent.com/pyrall0712/Eksperimen_SML_MuhamadRizal/main/preprocessing/namadataset_preprocessing/iris_test_processed.csv"
    
    try:
        train_df = pd.read_csv(url_train).dropna()
        test_df = pd.read_csv(url_test).dropna()
    except Exception as e:
        print(f"Gagal mengambil data dari GitHub: {str(e)}")
        return

    X_train = train_df.drop(columns=['Species'])
    y_train = train_df['Species']
    X_test = test_df.drop(columns=['Species'])
    y_test = test_df['Species']

    # 3. Mengunci nama eksperimen agar sinkron dengan DagsHub
    mlflow.set_experiment("Iris_Classification_Baseline")

    # Menggunakan nama run yang spesifik agar mudah dilacak di Artifacts
    with mlflow.start_run(run_name="CI_Automated_Run"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Log parameter dan metrik ke DagsHub
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        
        # PENTING: Perintah wajib agar folder artefak model fisik terunggah ke DagsHub
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            registered_model_name="Iris_RandomForest_Model"
        )
        
        print(f"🚀 Berhasil! Akurasi Model CI: {acc:.4f}")
        print("Artefak model telah sukses dikirim ke DagsHub.")

if __name__ == "__main__":
    train_model()