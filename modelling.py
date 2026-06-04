import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def train_model():
    token = os.getenv("DAGSHUB_TOKEN_BYPASS")
    
    if token:
        print("🔧 Menghubungkan MLflow langsung ke Remote Tracker DagsHub...")
        os.environ["MLFLOW_TRACKING_USERNAME"] = "pyrall0712"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        
        repo_owner = "pyrall0712"
        repo_name = "Eksperimen_SML_MuhamadRizal"
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

    url_train = "https://raw.githubusercontent.com/pyrall0712/Eksperimen_SML_MuhamadRizal/main/preprocessing/namadataset_preprocessing/iris_train_processed.csv"
    url_test = "https://raw.githubusercontent.com/pyrall0712/Eksperimen_SML_MuhamadRizal/main/preprocessing/namadataset_preprocessing/iris_test_processed.csv"
    
    train_df = pd.read_csv(url_train).dropna()
    test_df = pd.read_csv(url_test).dropna()

    X_train = train_df.drop(columns=['Species'])
    y_train = train_df['Species']
    X_test = test_df.drop(columns=['Species'])
    y_test = test_df['Species']

    mlflow.set_experiment("Iris_Classification_Baseline")

    with mlflow.start_run(run_name="CI_Automated_Run"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            registered_model_name="Iris_RandomForest_Model"
        )
        print(f"🚀 Berhasil! Akurasi Model CI: {acc:.4f}")

if __name__ == "__main__":
    train_model()