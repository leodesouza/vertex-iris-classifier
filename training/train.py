import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from google.cloud import storage

def train():
    # AIP_MODEL_DIR: onde o Vertex espera o modelo
    # AIP_CHECKPOINT_DIR: podemos usar para exportar dados auxiliares
    gcs_output_path = os.environ.get("AIP_MODEL_DIR")
    
    # 1. Dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # 2. Modelo
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # 3. Exportação Local Temporária
    os.makedirs("outputs", exist_ok=True)
    local_model_path = "outputs/model.joblib"
    local_test_path = "outputs/test_data.csv"
    
    joblib.dump(model, local_model_path)
    
    # Criamos um DataFrame para o teste (incluindo a label para o avaliador)
    test_df = pd.DataFrame(X_test, columns=iris.feature_names)
    test_df['target'] = y_test
    test_df.to_csv(local_test_path, index=False)
    
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
        
        # Upload do Dataset de Teste (Para o componente de Evaluate ler)
        # Salvamos em uma subpasta /artifacts/ para organização
        test_blob = bucket.blob(f"{blob_prefix}artifacts/test_data.csv")
        test_blob.upload_from_filename(local_test_path)
        print(f"Dados de teste enviados para: {gcs_output_path}artifacts/test_data.csv")

if __name__ == "__main__":
    train()