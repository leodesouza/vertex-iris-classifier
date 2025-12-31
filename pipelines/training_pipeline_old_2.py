from kfp import dsl
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.types import artifact_types
# from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from pipelines.components.evaluate import evaluate_model

# Definição de constantes para facilitar a manutenção
PROJECT_ID = "certifiedgpt"
REGION = "us-central1"
BUCKET_NAME = "certifiedgpt-vertex-pipelines-us-central1"
# O diretório base onde os artefatos serão salvos
MODEL_GCS_DIR = f"gs://{BUCKET_NAME}/models/iris-model"
# Imagem que você buildou com o Dockerfile e train.py atualizados
TRAIN_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-containers/vertex-iris-expert-pipeline:latest"

@dsl.pipeline(name="iris-expert-with-gate")
def iris_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    base_output_dir: str = MODEL_GCS_DIR
):
    # 1. TREINO - Perfeito, mantém a máquina n1-standard-4
    train_task = CustomTrainingJobOp(
        project=project,
        location=location,
        display_name="train-job",
        base_output_directory=base_output_dir,
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {"image_uri": TRAIN_IMAGE}
        }],
    )
    
    # train_task.set_cpu_limit('4')
    # train_task.set_memory_limit('16G')
    # train_task.add_node_selector_constraint('nvidia-tesla-t4')


    # 2. AVALIAÇÃO
    eval_task = evaluate_model(
        model_gcs_path=f"{base_output_dir}/model",
        test_dataset=f"{MODEL_GCS_DIR}/artifacts/test_data.csv"
    )
    eval_task.after(train_task)
    
    # 3. CONDIÇÃO DE DEPLOY (Unificando o Gate aqui)
    # Substituí o gate_task pela Condition para simplificar o grafo
    with dsl.Condition(eval_task.outputs["accuracy"] >= 0.90, name="min-threshold-check"):
             
        import_model_task = dsl.importer(
            artifact_uri=f"{base_output_dir}/model",
            artifact_class=dsl.Artifact,
            metadata={
                "containerSpec": {
                    "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
                }
            }
        )
           
        # 2. Agora o ModelUploadOp recebe o output do importer
        upload_task = ModelUploadOp(
            project=project,
            location=location,
            display_name="iris-expert-model",
            unmanaged_container_model=import_model_task.output
        )

        
        upload_task.after(eval_task)
        # # ENDPOINT
        # endpoint_task = EndpointCreateOp(
        #     project=project, 
        #     location=location,
        #     display_name="iris-endpoint"
        # )

        # # DEPLOY
        # ModelDeployOp(
        #     model=upload_task.outputs["model"],
        #     endpoint=endpoint_task.outputs["endpoint"],
        #     dedicated_resources_machine_type="n1-standard-2",
        #     dedicated_resources_min_replica_count=1
        # )