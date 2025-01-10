# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import mlflow.sklearn
import pandas as pd

app = FastAPI()

class GenderEnum(str, Enum):
    Male = "Male"
    Female = "Female"

class MultipleLinesEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    No_Phone_Service = "No phone service"

class InternetServiceEnum(str, Enum):
    DSL = "DSL"
    Fiber_Optic = "Fiber optic"
    No = "No"

class OnlineSecurity(str, Enum):
    No = 'No'
    Yes = 'Yes'
    No_Internet_Service = 'No internet service'

class OnlineBackupEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    No_Internet_Service = "No internet service"

class DeviceProtectionEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    No_Internet_Service = "No internet service"

class TechSupportEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    No_Internet_Service = "No internet service"

class StreamingTVEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    No_Internet_Service = "No internet service"

class StreamingMoviesEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    No_Internet_Service = "No internet service"

class ContractEnum(str, Enum):
    Month_to_Month = "Month-to-month"
    One_Year = "One year"
    Two_Year = "Two year"

class PaperlessBillingEnum(str, Enum):
    Yes = "Yes"
    No = "No"

class PaymentMethodEnum(str, Enum):
    Electronic_Check = "Electronic check"
    Mailed_Check = "Mailed check"
    Bank_Transfer = "Bank transfer (automatic)"
    Credit_Card = "Credit card (automatic)"

class PredictionRequest(BaseModel):
    gender: GenderEnum
    SeniorCitizen: bool
    Partner: bool
    Dependents: bool
    tenure: int
    PhoneService: bool
    MultipleLines: MultipleLinesEnum
    InternetService: InternetServiceEnum
    OnlineSecurity: OnlineSecurity
    OnlineBackup: OnlineBackupEnum
    DeviceProtection: DeviceProtectionEnum
    TechSupport: TechSupportEnum
    StreamingTV: StreamingTVEnum
    StreamingMovies: StreamingMoviesEnum
    Contract: ContractEnum
    PaperlessBilling: PaperlessBillingEnum
    PaymentMethod: PaymentMethodEnum
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    prediction: bool

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # mlflow.set_tracking_uri("http://mlflow:5000")

        # model_name = "sk-learn-random-forest-reg-model"
        # model_version = "latest"

        # model_uri = f"models:/{model_name}/{model_version}"
        # model = mlflow.sklearn.load_model(model_uri)

        print("Model loaded successfully.")

        # Convert request data to DataFrame
        df = pd.DataFrame([request.dict()])

        # Preprocess the input data
        df = df.dropna()

        gender_mapping = {"Male": 0, "Female": 1}
        df["gender"] = df["gender"].map(gender_mapping)

        binary_mapping = {"No": 0, "Yes": 1}
        binary_col = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col_name in binary_col:
            df[col_name] = df[col_name].map(binary_mapping)

        categorical_col = [
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
        ]
        for col_name in categorical_col:
            df[col_name] = df[col_name].astype("category").cat.codes

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        print(df.head())

        return PredictionResponse(prediction=True)

        # predictions = model.predict(input_data)
        # return PredictionResponse(prediction=bool(predictions[0]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
