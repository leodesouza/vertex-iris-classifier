# 🌸 Iris Expert MLOps Pipeline (Vertex AI)

Este repositório contém a implementação de um pipeline de Machine Learning de nível **Especialista** para a classificação do dataset Iris. O projeto utiliza **Google Cloud Vertex AI Pipelines**, **Kubeflow Pipelines (KFP)** e foca em práticas rigorosas de **MLOps**, como linhagem de artefatos e validação de performance.

---

## 🚀 Arquitetura do Pipeline

Diferente de pipelines básicos, esta implementação foca em **Governança** e **Qualidade**. O fluxo de trabalho automatizado segue estas etapas:



1.  **Custom Training**: Executa o treinamento em um container Docker isolado, salvando o modelo no GCS.
2.  **Model Evaluation**: O componente de avaliação gera métricas visuais (Matriz de Confusão e Curva ROC) que ficam integradas ao console do Vertex AI.
3.  **Performance Gate (Champion vs Challenger)**: 
    * O pipeline busca o modelo atual em produção.
    * Compara a acurácia do novo modelo (Challenger) com o atual (Champion).
    * **O pipeline é interrompido com erro** se o novo modelo for inferior, impedindo deploys ruins.
4.  **Model Registry**: Registro oficial e versionamento do modelo aprovado.
5.  **Online Serving**: Criação de um Endpoint e deploy automático para consumo via API.

---

## 📂 Estrutura de Pastas

```text
vertex-iris-expert/
├── components/              # Componentes leves baseados em função Python
│   ├── evaluate.py          # Gera métricas e visualizações (ROC/CM)
│   └── performance_check.py  # O "Juiz" (Performance Gate)
├── src/                     # Código que roda dentro do Container
│   ├── train.py             # Script de treinamento principal
│   ├── requirements.txt     # Dependências do container de treino
│   └── Dockerfile           # Definição da imagem Docker
├── pipelines/
│   └── pipeline.py          # Definição do Grafo (DAG) do pipeline
├── run_pipeline.py          # Script para compilar e disparar o job no GCP
├── predict_test.py          # Script de teste de predição no Endpoint
└── .gitignore               # Arquivos ignorados pelo Git