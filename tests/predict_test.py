from google.cloud import aiplatform

def predict_iris_test():
    # 1. Configurações (Ajuste o ENDPOINT_ID se necessário)
    PROJECT_ID = "certifiedgpt"
    REGION = "us-central1"
    
    # Você encontra o ID no console em Vertex AI > Endpoints
    # Ou pode buscar o mais recente via código como abaixo
    ENDPOINT_NAME = "iris-production-endpoint"

    aiplatform.init(project=PROJECT_ID, location=REGION)

    # 2. Localizar o Endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_NAME}"', 
        order_by="create_time desc"
    )

    if not endpoints:
        print("Endpoint não encontrado!")
        return

    endpoint = endpoints[0]
    print(f"Conectado ao Endpoint: {endpoint.resource_name}")

    # 3. Dados para Predição (Formato esperado pelo Scikit-Learn: Lista de Listas)
    # Exemplo: [comprimento_sépala, largura_sépala, comprimento_pétala, largura_pétala]
    instances = [
        [5.1, 3.5, 1.4, 0.2],  # Esperado: Setosa (Classe 0)
        [6.7, 3.1, 4.7, 1.5],  # Esperado: Versicolor (Classe 1)
    ]

    # 4. Chamar a API de Predição
    response = endpoint.predict(instances=instances)

    # 5. Exibir Resultados
    print("\n--- Resultados da Predição ---")
    for i, prediction in enumerate(response.predictions):
        # O retorno é o índice da classe (0, 1 ou 2)
        class_id = int(prediction)
        target_names = ['Setosa', 'Versicolor', 'Virginica']
        print(f"Instância {i+1}: Classe {class_id} ({target_names[class_id]})")

if __name__ == "__main__":
    predict_iris_test()

# para rodar 
# gcloud auth application-default login
# python predict_test.py    