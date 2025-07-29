import sys
import os
import pickle
sys.path.append('/opt/airflow')

from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    description='ML pipeline using dag and task decorators for UITF data',
    schedule=None, 
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'taskflow'],
)

def get_data(**context):
    train_data, test_data = preprocess_data()
    
    os.makedirs('/tmp/airflow_data', exist_ok=True)
    
    with open('/tmp/airflow_data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open('/tmp/airflow_data/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    context['task_instance'].xcom_push(key='train_data_path', value='/tmp/airflow_data/train_data.pkl')
    context['task_instance'].xcom_push(key='test_data_path', value='/tmp/airflow_data/test_data.pkl')

def model_training(**context):
    train_data_path = context['task_instance'].xcom_pull(key='train_data_path', task_ids='get_data')
    
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    
    model = train_model(train_data)
    
    model_path = '/tmp/airflow_data/model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path

def model_evaluation(**context):
    model_path = context['task_instance'].xcom_pull(task_ids='model_training')
    test_data_path = context['task_instance'].xcom_pull(key='test_data_path', task_ids='get_data')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    scores = evaluate_model(model, test_data)
    print(scores)
    return scores

get_data_task = PythonOperator(
    task_id='get_data',
    python_callable=get_data,
    dag=dag,
)

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag,
)

model_evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation,
    dag=dag,
)

get_data_task >> model_training_task >> model_evaluation_task
