from kfp import compiler
from google.cloud import aiplatform
# Ajustado para refletir sua nova estrutura de pastas
from pipelines.training_pipeline import iris_pipeline 

def run():
    # Centralizando as configurações
    PROJECT_ID = "certifiedgpt"
    REGION = "us-central1"
    BUCKET_NAME = "certifiedgpt-vertex-pipelines-us-central1"
    PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"

    aiplatform.init(project=PROJECT_ID, location=REGION)

    # 1. Compilação
    # Gera o arquivo JSON que descreve o grafo do Kubernetes
    package_path = "expert_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path=package_path
    )

    # 2. Execução no Vertex AI
    job = aiplatform.PipelineJob(
        display_name="expert-iris-run",
        template_path=package_path,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project": PROJECT_ID,
            "location": REGION,
            "base_output_dir": f"gs://{BUCKET_NAME}/models/iris-model"
        },
        enable_caching=True # Mantém o cache para economizar tempo/dinheiro em steps que não mudaram
    )

    # submit() envia para a nuvem; submit(service_account=...) seria o ideal para produção
    job.submit()

if __name__ == "__main__":
    run()