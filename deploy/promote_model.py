from google.cloud import aiplatform

def deploy_model_to_production(model_id: str, endpoint_id: str = None):
    aiplatform.init(project="certifiedgpt", location="us-central1")

    # 1. Referenciar o modelo já registrado no Registry
    model = aiplatform.Model(model_name=model_id)

    # 2. Obter ou Criar um Endpoint
    if endpoint_id:
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
    else:
        endpoint = aiplatform.Endpoint.create(display_name="iris-prod-endpoint")

    # 3. O Deploy Real (Onde o custo de infraestrutura começa)
    print("Iniciando o deploy do modelo aprovado...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="iris-v1-approved",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=2,
        # Aqui você pode definir o tráfego. 100 significa que ele assume tudo.
        traffic_split={"0": 100} 
    )
    print(f"Modelo disponível em: {endpoint.resource_name}")

if __name__ == "__main__":
    # O ID do modelo você pega no console do Vertex AI ou no output do seu pipeline
    MY_MODEL_ID = "projects/your-project/locations/us-central1/models/your-model-id"
    deploy_model_to_production(model_id=MY_MODEL_ID)