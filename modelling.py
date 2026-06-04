import os
import shutil
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def train_model():
    token = os.getenv("DAGSHUB_TOKEN_BYPASS")
    repo_owner = "pyrall0712"
    repo_name = "Eksperimen_SML_MuhamadRizal"
    
    if token:
        print("🔧 Menghubungkan MLflow langsung ke Remote Tracker DagsHub...")
        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

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

    with mlflow.start_run(run_name="CI_Automated_Run") as run:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        
        # 1. Simpan model ke dalam folder lokal sementara server GitHub terlebih dahulu
        local_model_dir = "temp_model_folder"
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)
            
        print("💾 Menyimpan model ke direktori lokal server...")
        mlflow.sklearn.save_model(sk_model=model, path=local_model_dir)
        
        # 2. TRIK KHUSUS: Paksa unggah folder lokal tadi langsung ke server DagsHub
        print("🚀 Memaksa unggah folder model secara direct ke DagsHub Artifacts...")
        mlflow.log_artifacts(local_dir=local_model_dir, artifact_path="model")
        
        # Bersihkan folder lokal sementara setelah sukses diunggah
        shutil.rmtree(local_model_dir)
        
        print(f"🎯 Selesai! Akurasi Model CI: {acc:.4f}")

if __name__ == "__main__":
    train_model()