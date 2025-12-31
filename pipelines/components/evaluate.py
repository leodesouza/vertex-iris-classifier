from typing import NamedTuple
from kfp.dsl import component, Input, Output, Model, Metrics, ClassificationMetrics

@component(
    base_image="python:3.10-slim",
    packages_to_install=["scikit-learn", "joblib", "pandas"]
)
def evaluate_model(
    model_artifact: Input[Model],
    test_dataset: str, # Recebe o caminho gs://.../test_data.csv
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics]
) -> NamedTuple("Outputs", [("accuracy", float)]):
    import joblib
    import pandas as pd
    import os
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
    from collections import namedtuple

    # 1. Carregar Modelo
    model_path = model_artifact.path
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.joblib")
    model = joblib.load(model_path)
    
    # 2. Carregar Dados de Teste reais do GCS (produzidos pelo treino)
    df = pd.read_csv(test_dataset)
    X_test = df.drop(columns=['target'])
    y_test = df['target']
    
    # 3. Predições
    predictions = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # 4. Métricas Escalares
    acc = float(accuracy_score(y_test, predictions))
    metrics.log_metric("accuracy", acc)
    
    # 5. Matriz de Confusão Visual
    cm = confusion_matrix(y_test, predictions)
    classification_metrics.log_confusion_matrix(
        categories=["setosa", "versicolor", "virginica"],
        matrix=cm.tolist()
    )
    
    # 6. Curva ROC (Exemplo para uma das classes)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
    classification_metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())

    # 7. Retorno para o Pipeline
    output = namedtuple("Outputs", ["accuracy"])
    return output(acc)