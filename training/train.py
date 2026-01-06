import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from google.cloud import storage
import argparse

def train():
    
    # AIP_MODEL_DIR: onde o Vertex espera o modelo
    # AIP_CHECKPOINT_DIR: podemos usar para exportar dados auxiliares
    BUCKET_NAME = "certifiedgpt-vertex-pipelines-us-central1"        
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()
    RUN_ID = args.run_id    
    gcs_output_path = f"gs://{BUCKET_NAME}/pipelines/iris/runs/{RUN_ID}/model/" 
    data_path = f"gs://{BUCKET_NAME}/pipelines/iris/runs/{RUN_ID}/data/"
    
    
    
    # 1. Dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    y_train = y_train.astype(int)
    # 2. Modelo
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # 3. Exportação Local Temporária
    os.makedirs("outputs", exist_ok=True)
    local_model_path = "outputs/model.joblib"        
    joblib.dump(model, local_model_path)
    
    test_df = pd.DataFrame(X_test)
    test_df.to_csv("outputs/test_no_target.csv", index=False, header=False)
    test_with_target = test_df.copy()
    test_with_target['target'] = y_test
    test_with_target['target'] = y_test.astype(str)
    test_with_target.to_csv("outputs/test_ground_truth.csv", index=False)
    
    
    # 4. Upload para GCS
    if gcs_output_path:
        # Extrair bucket e prefixo
        path_parts = gcs_output_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        blob_prefix = "/".join(path_parts[1:])
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Upload do Modelo (O Vertex espera isso para o registro)
        model_blob = bucket.blob(f"{blob_prefix}model.joblib")
        model_blob.upload_from_filename(local_model_path)
        print(f"Modelo enviado para: {gcs_output_path}model.joblib")  
                                        
        
    # 5 dataset
    
     # Extrair bucket e prefixo
        path_parts = data_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        blob_prefix = "/".join(path_parts[1:])
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
                                
        # Em vez de salvar na raiz:        
        bucket.blob(f"{blob_prefix}test_no_target.csv").upload_from_filename("outputs/test_no_target.csv")
        bucket.blob(f"{blob_prefix}test_ground_truth.csv").upload_from_filename("outputs/test_ground_truth.csv")
    
        
        

if __name__ == "__main__":
    train()