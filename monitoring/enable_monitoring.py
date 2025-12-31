from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring

# Configuração do alerta
alert_config = model_monitoring.ThresholdConfig(
    value=0.001 # Limite de desvio estatístico (Distância de Jensen-Shannon)
)

# Criando o Job de Monitoramento
monitor_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="iris-expert-monitoring",
    endpoint=endpoint, # O endpoint que criamos no pipeline
    user_emails=["seu-email@dominio.com"],
    feature_thresholds={"feature_name_1": 0.001}, # Monitora colunas específicas
    schedule_config=model_monitoring.ScheduleConfig(monitor_interval=1), # Monitora a cada 1 hora
)