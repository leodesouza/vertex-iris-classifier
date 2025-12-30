from kfp import dsl
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from components.evaluate import evaluate_model
from components.validade_model_performance import validate_performance_gate

@dsl.pipeline(name="iris-expert-with-gate")
def iris_pipeline(
    project: str = "certifiedgpt",
    region: str = "us-central1",
    bucket: str = "gs://certifiedgpt-vertex-pipelines-us-central1"
):
    # 1. TREINO
    train_task = CustomTrainingJobOp(...)

    # 2. AVALIAÇÃO (Gera as métricas visuais e o objeto metrics)
    eval_task = evaluate_model(
        model_artifact=train_task.outputs["model"]
    )

    # 3. GATE DE PERFORMANCE (O "Juiz")
    # Este step vai falhar o pipeline se a acurácia for baixa
    gate_task = validate_performance_gate(
        project=project,
        location=region,
        new_metrics=eval_task.outputs["metrics"],
        model_display_name="iris-expert-model"
    )

    # 4. REGISTRO E DEPLOY (Só ocorrem se o gate_task tiver sucesso)
    with dsl.Condition(eval_task.outputs["accuracy"] >= 0.90, name="min-threshold-check"):
        
        # O .after(gate_task) garante que só tentamos registrar se a comparação passar
        upload_task = ModelUploadOp(
            project=project,
            location=region,
            display_name="iris-expert-model",
            unmanaged_container_model=train_task.outputs["model"],
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
            labels={"accuracy": str(eval_task.outputs["accuracy"]).replace(".", "_")}
        ).after(gate_task)

        endpoint_task = EndpointCreateOp(project=project, location=region).after(upload_task)

        ModelDeployOp(
            model=upload_task.outputs["model"],
            endpoint=endpoint_task.outputs["endpoint"],
            dedicated_resources_machine_type="n1-standard-2",
            dedicated_resources_min_replica_count=1
        )