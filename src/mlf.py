import mlflow
import subprocess
import os
import configparser


config = configparser.ConfigParser()
config.read('config.ini') 
MLFLOW_TRACKING_URI= config.get('Default', 'MLFLOW_TRACKING_URI')

def start_mlflow_ui():
    local_mlflow_port = 5033
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow_ui_cmd = f"mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI} --port {local_mlflow_port}"
    mlflow_process = subprocess.Popen(mlflow_ui_cmd, shell=True)
    print(f"MLflow UI running at http://localhost:{local_mlflow_port}")

    try:
        mlflow_process.communicate()
    except KeyboardInterrupt:
        mlflow_process.terminate()
        print("MLflow UI arrêté")