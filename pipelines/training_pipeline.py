import json
from kfp import dsl
from kfp.dsl import OneOf
from kfp.dsl import component

from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.model_evaluation import ModelEvaluationRegressionOp, ModelEvaluationClassificationOp
from google_cloud_pipeline_components.types import artifact_types
from pipelines.components.evaluate import evaluate_model
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp

PROJECT_ID = "certifiedgpt"
REGION = "us-central1"


from kfp.dsl import component, Input


@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-pipeline-components"]
)
def debug_model(model: Input[artifact_types.VertexModel]):
    print("MODEL RESOURCE NAME =", model.metadata.get("resourceName"))
    print("MODEL URI =", model.uri)


# @component(base_image="python:3.10")
# def log_branch(message: str):
#     print(f"[PIPELINE BRANCH] {message}")

#pipeline_root=PIPELINE_ROOT
@dsl.pipeline(name="iris-pipeline")
def iris_pipeline(    
    project: str = PROJECT_ID,
    location: str = REGION,    
    existing_model: bool = False
):
  
    
    BUCKET_NAME = "certifiedgpt-vertex-pipelines-us-central1"    
    OUTPUT_DIRECTORY = f"gs://{BUCKET_NAME}/pipelines/iris/runs"
    BUCKET_URI = "gs://{}".format(BUCKET_NAME)
    TRAIN_IMAGE = (
        f"{REGION}-docker.pkg.dev/{PROJECT_ID}"
        "/ml-containers/vertex-iris-expert-pipeline:latest"
    )  
    EMAIL_RECIPIENTS = [ "leo.desouza@gmail.com" ]
    notify_task = VertexNotificationEmailOp(
                    recipients= EMAIL_RECIPIENTS
                    )
         
    with dsl.ExitHandler(notify_task, name='MLOps Continuous Training Pipeline'):
        
    
        custom_job = CustomTrainingJobOp(
            project=project,
            location=location,
            display_name="iris-train",
            base_output_directory=OUTPUT_DIRECTORY,
            worker_pool_specs=[
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-4"
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": TRAIN_IMAGE,
                        "args": [
                            "--run_id",
                            "{{$.pipeline_job_uuid}}"
                        ]
                    }
                }
            ],
        )
    
        import_unmanaged_model_task = dsl.importer(
            #artifact_uri="{{$.pipeline_root}}/{{$.pipeline_job_uuid}}",
            artifact_uri=f"{OUTPUT_DIRECTORY}/{{{{$.pipeline_job_uuid}}}}/model",            
            artifact_class=artifact_types.UnmanagedContainerModel,
            metadata={
                "containerSpec": {
                    "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
                }
            }
        ).after(custom_job)

        
        with dsl.If(existing_model == True, name="use-existing model"):            
            imported_model = dsl.importer(
                artifact_uri="unused",
                artifact_class=artifact_types.VertexModel,
                metadata={
                "resourceName": (
                    f"projects/202417564163/locations/{REGION}/models/2067529361451384832"
                ),
                },
             )

            batch_predict_existing = ModelBatchPredictOp(
                project=project,
                location=location,
                job_display_name="prediction-batch-existing",
                model=imported_model.outputs["artifact"],
                instances_format="csv",
                predictions_format="jsonl",
                gcs_source_uris=[
                    f"{OUTPUT_DIRECTORY}/{{{{$.pipeline_job_uuid}}}}/data/test_no_target.csv"
                ],
                gcs_destination_output_uri_prefix=f"{BUCKET_URI}/batch_predict_output",
                machine_type="n1-standard-2",
            )
            
        with dsl.Else(name="create-new model"):      
            model_upload_op = ModelUploadOp(
                project=project,
                location=location,
                display_name="iris-expert-model",               
                unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],                
            )
                                                
        # debug_model(model=model_upload_op.outputs["model"])

            batch_predict_else = ModelBatchPredictOp(
                project=project,
                location=location,
                job_display_name="prediction-batch-new-model",
                model=model_upload_op.outputs["model"],
                instances_format="csv",
                predictions_format="jsonl",
                gcs_source_uris=[
                    f"{OUTPUT_DIRECTORY}/{{{{$.pipeline_job_uuid}}}}/data/test_no_target.csv"
                ],
                gcs_destination_output_uri_prefix=f"{BUCKET_URI}/batch_predict_output",
                machine_type="n1-standard-2",
                )  
            
            evaluation_task_else = ModelEvaluationClassificationOp(
                project=project,
                location=location,
                target_field_name="target",
                # Em classificação, usamos 'prediction_label_column' para a classe final
                prediction_label_column="prediction", 
                prediction_score_column="prediction",
                predictions_format="jsonl",
                predictions_gcs_source=batch_predict_else.outputs["gcs_output_directory"],
                ground_truth_format="csv",
                class_labels=["0", "1", "2"],
                ground_truth_gcs_source=[f"{OUTPUT_DIRECTORY}/{{{{$.pipeline_job_uuid}}}}/data/test_ground_truth.csv"]
            )   
            
                                            
        # Evaluation task
        # evaluation_task = ModelEvaluationRegressionOp(
        #                     project= project,
        #                     target_field_name= "species",
        #                     location= location,
        #                     # model= model_resource,
        #                     prediction_score_column="prediction",
        #                     predictions_format= "jsonl",
        #                     predictions_gcs_source= batch_predict_task.outputs["gcs_output_directory"],
        #                     ground_truth_format= "csv",
        #                     ground_truth_gcs_source= [f"{BUCKET_URI}/test.csv"]
        #                     )
        
    return
    

    # ======================================================
    # 3. EVALUATION
    # ======================================================
    # eval_task = evaluate_model(
    #     model_gcs_path=f"{base_output_dir}/model",
    #     test_dataset=f"{base_output_dir}/model/artifacts/test_data.csv"
    # )


    # ======================================================
    # 4. ACCURACY GATE
    # ======================================================
    # with dsl.Condition(
    #     eval_task.outputs["accuracy"] >= 0.90,
    #     name="accuracy-gate"
    # ):

        # ==================================================
        # 5. REGISTER MODEL (Model Registry)
        # ==================================================
    
   
