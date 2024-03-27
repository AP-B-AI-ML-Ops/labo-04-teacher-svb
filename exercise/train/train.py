import os
import pickle
import mlflow

from sklearn.ensemble import RandomForestRegressor

from prefect import flow, task

@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def start_mlflow_run(X_train, y_train):
    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

@flow
def train_flow(models_path: str):
    mlflow.set_experiment("random-forest-train")
    mlflow.sklearn.autolog()

    X_train, y_train = load_pickle(os.path.join(models_path, "train.pkl"))

    start_mlflow_run(X_train, y_train)

