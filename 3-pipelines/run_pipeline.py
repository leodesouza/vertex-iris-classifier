from kfp import compiler
from google.cloud import aiplatform
from pipeline import iris_pipeline

def run():
    aiplatform.init(project="certifiedgpt", location="us-central1")

    # Compila
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path="expert_pipeline.json"
    )

    # Executa
    job = aiplatform.PipelineJob(
        display_name="expert-iris-run",
        template_path="expert_pipeline.json",
        pipeline_root="gs://certifiedgpt-vertex-pipelines-us-central1/artifacts",
        enable_caching=True
    )
    job.submit()

if __name__ == "__main__":
    run()