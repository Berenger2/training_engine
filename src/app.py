from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
import mlflow
import numpy as np
from mlflow import MlflowClient
import pandas as pd
from mlflow.entities import ViewType
import joblib
import os
import configparser


config = configparser.ConfigParser()
config.read('config.ini') 
MLFLOW_TRACKING_URI= config.get('Default', 'MLFLOW_TRACKING_URI')

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


app = FastAPI(
    title="FastAPI MLFlow",
    description="FastAPI application with MLFlow.",
    version="0.1",
    docs_url="/docs",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/experiences", tags=["Mlflow"])
async def get_current_time():
    experiments = client.search_experiments()
    response = []
    for experiment in experiments:
        experiment_info = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "artifact_location": experiment.artifact_location,
        }
        response.append(experiment_info)

    return JSONResponse(content=response)


@app.get("/runs", tags=["Mlflow"])
async def get_runs():
    runs = client.search_runs(ViewType.ACTIVE_ONLY)
    response = []
    for run in runs:
        run_info = {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                }
        response.append(run_info)

    return JSONResponse(content=response)

@app.get("/models", tags=["Mlflow"])
async def get_models():
    registered_models = client.search_registered_models()
    response = []
    for model in registered_models:
        model_info = {
            "name": model.name,
            "versions": []
        }
        for version in model.latest_versions:
            version_info = {
                "version": version.version,
                "run_id": version.run_id
            }
            model_info["versions"].append(version_info)
        response.append(model_info)
    return JSONResponse(content=response)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
