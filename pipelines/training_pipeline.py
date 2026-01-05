import json
from kfp import dsl
from kfp.dsl import OneOf
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.model_evaluation import ModelEvaluationRegressionOp, ModelEvaluationClassificationOp
from google_cloud_pipeline_components.types import artifact_types
from pipelines.components.evaluate import evaluate_model
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp

PROJECT_ID = "certifiedgpt"
REGION = "us-central1"

#pipeline_root=PIPELINE_ROOT
@dsl.pipeline(name="iris-pipeline")
def iris_pipeline(    
    project: str = PROJECT_ID,
    location: str = REGION,    
    existing_model: bool=False    
):
  
    
    BUCKET_NAME = "certifiedgpt-vertex-pipelines-us-central1"
    #PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipelines/iris/runs"
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

        
        with dsl.If(existing_model == True):
            # Import the parent model to upload as a version
            import_registry_model_task = dsl.importer(
                                        #artifact_uri="{{$.pipeline_root}}/{{$.pipeline_job_uuid}}",
                                        artifact_uri=f"{OUTPUT_DIRECTORY}/{{{{$.pipeline_job_uuid}}}}/model",
                                        artifact_class=artifact_types.VertexModel,
                                        metadata={
                                            "resourceName": f"projects/{PROJECT_ID}/locations/{REGION}/models/1234567890123472",
                                        },
                                    ).after(import_unmanaged_model_task)
            # Upload the model as a version
            model_version_upload_op = ModelUploadOp(
                                    project=project,
                                    location=location,
                                    display_name="model_display_name",
                                    parent_model=import_registry_model_task.outputs["artifact"],
                                    unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
                                )
        with dsl.Else():            
            model_upload_op = ModelUploadOp(
                project=project,
                location=location,
                display_name="iris-expert-model",        
                unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"]            
            )
        
        model_resource = dsl.OneOf(
            model_upload_op.outputs["model"], 
            model_version_upload_op.outputs["model"]
        )
        # Batch prediction
        batch_predict_task = ModelBatchPredictOp(
                            project= project,
                            job_display_name= "prediction-batch-job",
                            model=model_resource,
                            location= location,
                            instances_format= "csv",
                            predictions_format= "jsonl",
                            gcs_source_uris= [f"{BUCKET_URI}/test_no_target.csv"],
                            gcs_destination_output_uri_prefix= f"{BUCKET_URI}/batch_predict_output",
                            machine_type= 'n1-standard-4'
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
        evaluation_task = ModelEvaluationClassificationOp(
                project=project,
                location=location,
                target_field_name="target",
                # Em classificação, usamos 'prediction_label_column' para a classe final
                prediction_label_column="prediction", 
                prediction_score_column="prediction",
                predictions_format="jsonl",
                predictions_gcs_source=batch_predict_task.outputs["gcs_output_directory"],
                ground_truth_format="csv",
                class_labels=["0", "1", "2"],
                ground_truth_gcs_source=[f"{BUCKET_URI}/test_ground_truth.csv"]
            )
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
    
   
