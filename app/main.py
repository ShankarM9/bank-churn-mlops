from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Bank Churn Prediction API")

# Define input data schema
class CustomerData(BaseModel):
    CreditScore: int
    Geography: int  # Encoded
    Gender: int     # Encoded
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Load Model (In production, load specific version or 'Production' stage)
# For local demo, we assume the model name "BankChurnModel" is registered
model_name = "BankChurnModel"
try:
    # Attempt to load latest version
    model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
except:
    print("Model not found in registry. Ensure train.py ran successfully.")
    model = None

@app.post("/predict")
def predict_churn(data: CustomerData):
    if not model:
        return {"error": "Model not loaded"}
    
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else 0
    
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability)
    }

# Run with: uvicorn app.main:app --reload