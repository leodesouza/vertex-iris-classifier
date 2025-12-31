from kfp.dsl import component, Input, Metrics

@component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-aiplatform"]
)
def validate_performance_gate(
    project: str,
    location: str,
    new_metrics: Input[Metrics],
    model_display_name: str,
    threshold: float = 0.0  # Margem mínima de melhoria se desejar
):
    from google.cloud import aiplatform
    import logging
    
    aiplatform.init(project=project, location=location)
    
    # 1. Recupera a acurácia do modelo que acabamos de treinar
    new_acc = float(new_metrics.metadata.get("accuracy", 0))
    logging.info(f"Acurácia do Novo Modelo (Challenger): {new_acc}")

    # 2. Busca a versão atual (Champion) no Model Registry
    models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    
    current_acc = 0.0
    if models:
        # Pega a versão mais recente e extrai a acurácia dos labels ou metadados
        # Nota: Para isso funcionar, o registro anterior deve ter salvo a acurácia nos labels
        champion_model = models[0]
        current_acc = float(champion_model.labels.get("accuracy", "0").replace("_", "."))
        logging.info(f"Acurácia do Modelo Atual (Champion): {current_acc}")
    else:
        logging.info("Nenhum modelo anterior encontrado. Primeira execução.")

    # 3. Lógica de Decisão
    if new_acc < (current_acc + threshold):
        error_msg = (f"FALHA NO GATE: O modelo novo ({new_acc}) não superou "
                     f"o modelo atual ({current_acc}). Interrompendo pipeline.")
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    
    logging.info("SUCESSO: Modelo aprovado para promoção!")