from prefect import task, flow
from preprocess import preprocess_data
from train import train_models

@task
def task_preprocess():
    print("Starting Preprocessing...")
    preprocess_data()

@task
def task_train():
    print("Starting Training...")
    train_models()

@flow(name="Churn Prediction Pipeline")
def main_flow():
    task_preprocess()
    task_train()

if __name__ == "__main__":
    main_flow()