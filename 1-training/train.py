import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from google.cloud import storage

def train():
    # AIP_MODEL_DIR é injetado automaticamente pelo Vertex AI
    gcs_output_path = os.environ.get("AIP_MODEL_DIR")
    
    # Dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Exportação Local
    os.makedirs("model_dir", exist_ok=True)
    local_path = "model_dir/model.joblib"
    joblib.dump(model, local_path)
    
    # Upload para GCS
    if gcs_output_path:
        bucket_name = gcs_output_path.replace("gs://", "").split("/")[0]
        blob_prefix = "/".join(gcs_output_path.replace("gs://", "").split("/")[1:])
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"{blob_prefix}/model.joblib".replace("//", "/"))
        blob.upload_from_filename(local_path)

if __name__ == "__main__":
    train()