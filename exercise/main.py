from prefect import flow
from load.collect import *
from load.prep import *
from train.hpo import *
from train.train import *
from train.register import *

import mlflow

@flow
def main_flow():
    print("started main flow")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    collect_flow("./data/")
    prep_flow("./data/", "./models/")

    train_flow("./models/")
    hpo_flow("./models/", 5)
    register_flow("./models/", 5)


if __name__ == "__main__":
    main_flow()