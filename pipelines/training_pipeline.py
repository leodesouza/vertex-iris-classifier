from kfp import dsl
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.types import artifact_types
from pipelines.components.evaluate import evaluate_model

PROJECT_ID = "certifiedgpt"
REGION = "us-central1"
BUCKET_NAME = "certifiedgpt-vertex-pipelines-us-central1"
MODEL_GCS_DIR = f"gs://{BUCKET_NAME}/models/iris-model"

TRAIN_IMAGE = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}"
    "/ml-containers/vertex-iris-expert-pipeline:latest"
)

@dsl.pipeline(name="iris-expert-with-gate")
def iris_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    base_output_dir: str = MODEL_GCS_DIR
):

    # ======================================================
    # 1. TRAINING
    # ======================================================
    train_task = CustomTrainingJobOp(
        project=project,
        location=location,
        display_name="iris-train",
        base_output_directory=base_output_dir,
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {"image_uri": TRAIN_IMAGE}
        }],
    )

    # ======================================================
    # 2. FORMALIZA O MODELO COMO ARTIFACT
    # ======================================================
    trained_model = dsl.importer(
        artifact_uri=f"{base_output_dir}/model",
        artifact_class=artifact_types.Model,
        metadata={
            "containerSpec": {
                "imageUri": (
                    "us-docker.pkg.dev/vertex-ai/"
                    "prediction/sklearn-cpu.1-3:latest"
                )
            }
        }
    ).after(train_task)

    # ======================================================
    # 3. EVALUATION
    # ======================================================
    eval_task = evaluate_model(
        model_gcs_path=f"{base_output_dir}/model",
        test_dataset=f"{base_output_dir}/model/artifacts/test_data.csv"
    )


    # ======================================================
    # 4. ACCURACY GATE
    # ======================================================
    with dsl.Condition(
        eval_task.outputs["accuracy"] >= 0.90,
        name="accuracy-gate"
    ):

        # ==================================================
        # 5. REGISTER MODEL (Model Registry)
        # ==================================================
        ModelUploadOp(
            project=project,
            location=location,
            display_name="iris-expert-model",
            unmanaged_container_model=trained_model.output
        ).after(eval_task)
