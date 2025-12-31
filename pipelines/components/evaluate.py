from typing import NamedTuple
from kfp.dsl import component, Output, Metrics, ClassificationMetrics

@component(
    base_image="python:3.10-slim",
    packages_to_install=["scikit-learn", "joblib", "pandas", "google-cloud-storage"]
)
def evaluate_model(
    model_gcs_path: str, # Mudamos de Input[Model] para str
    test_dataset: str,
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics]
) -> NamedTuple("Outputs", [("accuracy", float)]):
    import joblib
    import pandas as pd
    import os
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
    from collections import namedtuple
    from google.cloud import storage

    # 1. Download do modelo do GCS para o container local
    # Como não estamos usando Input[Model], baixamos manualmente
    model_local_path = "model.joblib"
    
    if model_gcs_path.startswith("gs://"):
        bucket_name = model_gcs_path.split("/")[2]
        blob_path = "/".join(model_gcs_path.split("/")[3:])
        # Garante que aponta para o arquivo e não para a pasta
        if not blob_path.endswith(".joblib"):
             blob_path = os.path.join(blob_path, "model.joblib")
             
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(model_local_path)
    
    model = joblib.load(model_local_path)
    
    # 2. Carregar Dados de Teste
    df = pd.read_csv(test_dataset)
    X_test = df.drop(columns=['target'])
    y_test = df['target']
    
    # 3. Predições e Métricas
    predictions = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, predictions))
    metrics.log_metric("accuracy", acc)
    
    # ... (restante do código de matriz de confusão e ROC igual) ...
    
    output = namedtuple("Outputs", ["accuracy"])
    return output(acc)