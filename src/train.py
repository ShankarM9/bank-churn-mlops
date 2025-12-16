import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

params = yaml.safe_load(open("params.yaml"))

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return accuracy, precision, recall, f1

def train_models():
    # Load Data
    df = pd.read_csv(params['data']['processed_path'])
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['data']['test_size'], random_state=params['data']['random_state']
    )

    # Define models to experiment with
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC()
    }

    mlflow.set_experiment("Bank_Churn_Prediction")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            (acc, prec, rec, f1) = eval_metrics(y_test, y_pred)
            
            # Log Metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            
            # Log Model
            mlflow.sklearn.log_model(model, "model")
            
            # Register Model (Optional: Registers the RF model as production candidate)
            if model_name == "RandomForest":
                mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", params['model']['name'])

            print(f"{model_name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    train_models()