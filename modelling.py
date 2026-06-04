import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def train_model():
    token = os.getenv("DAGSHUB_TOKEN_BYPASS")
    repo_owner = "pyrall0712"
    repo_name = "Eksperimen_SML_MuhamadRizal"
    
    if token:
        print("🔧 Mengonfigurasi MLflow Terpusat dengan Target S3 Storage DagsHub...")
        # Kredensial untuk Akses Log Teks
        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
        
        # Kredensial Wajib untuk Akses File Fisik (S3 Storage)
        os.environ["AWS_ACCESS_KEY_ID"] = repo_owner
        os.environ["AWS_SECRET_ACCESS_KEY"] = token
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"https://dagshub.com/{repo_owner}/{repo_name}.s3"

    # Mengambil data Iris hasil preprocessing dari repositori utama Anda
    url_train = "https://raw.githubusercontent.com/pyrall0712/Eksperimen_SML_MuhamadRizal/main/preprocessing/namadataset_preprocessing/iris_train_processed.csv"
    url_test = "https://raw.githubusercontent.com/pyrall0712/Eksperimen_SML_MuhamadRizal/main/preprocessing/namadataset_preprocessing/iris_test_processed.csv"
    
    train_df = pd.read_csv(url_train).dropna()
    test_df = pd.read_csv(url_test).dropna()

    X_train = train_df.drop(columns=['Species'])
    y_train = train_df['Species']
    X_test = test_df.drop(columns=['Species'])
    y_test = test_df['Species']

    mlflow.set_experiment("Iris_Classification_Baseline")

    # KUNCI UTAMA: Tentukan langsung tempat penyimpanan fisik ke s3 DagsHub
    remote_artifact_uri = f"s3://{repo_name}/artifacts"

    # Jalankan run dengan mengunci target lokasi artefaknya ke remote_artifact_uri
    with mlflow.start_run(run_name="CI_Automated_Run", artifact_location=remote_artifact_uri):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        
        print("📦 Memulai proses unggah berkas model biner secara langsung ke S3 DagsHub...")
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            registered_model_name="Iris_RandomForest_Model"
        )
        print(f"🚀 Berhasil! Akurasi Model CI: {acc:.4f}")

if __name__ == "__main__":
    train_model()