import sys
sys.path.append('/opt/airflow')


from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from datetime import datetime, timedelta
from airflow.decorators import task,dag

#activate environmentls
#C:/Users/Admin/Documents/GitHub/a2ed0f6138ae9c26927ed247300a0ce53787796d1d75de09ce5e1e3edf95e367_UITF_BUY_CLASSIFIER/.venv/Scripts/activate.bat actviate env

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    description='ML pipeline using dag and task decorators for UITF data',
    schedule_interval=None, 
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'taskflow'],
)

@task
def get_data():
    train_data, test_data = preprocess_data()
    return {'train': train_data, 'test': test_data}

@task
def model_training(train_data):
    model = train_model(train_data)
    return model

@task
def model_evaluation(model, test_data):
    scores = evaluate_model(model, test_data)
    print(f"Model scores: {scores}")
    return scores

def ml_pipeline():
    data = get_data()
    model = model_training(data['train'])
    model_evaluation(model, data['test'])
    
# Instantiate the DAG
dag_instance = ml_pipeline()