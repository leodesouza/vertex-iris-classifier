from kfp.dsl import component, Input, Output, Model, Metrics, ClassificationMetrics

@component(
    base_image="python:3.10-slim",
    packages_to_install=["scikit-learn", "google-cloud-storage", "joblib", "pandas"]
)
def evaluate_model(
    model_artifact: Input[Model],
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics]
):
    import joblib
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
    from sklearn.model_selection import train_test_split
    
    # 1. Carregar Modelo
    model = joblib.load(model_artifact.path + "/model.joblib")
    
    # 2. Preparar Dados de Teste
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # 3. Predições
    predictions = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1] # Para curva ROC
    
    # 4. Métricas Escalares
    acc = accuracy_score(y_test, predictions)
    metrics.log_metric("accuracy", float(acc))
    
    # 5. Matriz de Confusão Visual
    cm = confusion_matrix(y_test, predictions)
    classification_metrics.log_confusion_matrix(
        categories=iris.target_names.tolist(),
        matrix=cm.tolist()
    )
    
    # 6. Curva ROC (Exemplo para classe 1 vs resto)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
    classification_metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())