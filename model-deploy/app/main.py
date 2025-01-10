from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import joblib
import boto3
import tempfile
import os

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
    PaperlessBilling: bool
    PaymentMethod: PaymentMethodEnum
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    prediction: bool

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        df = preprocess_input(request)
        model = get_model()
        predictions = model.predict(df)
        return PredictionResponse(prediction=bool(predictions[0]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def get_model():
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin'
    )

    bucket_name = 'mlflow-artifacts'
    last_used_model_obj = s3.get_object(Bucket=bucket_name, Key='last_used_artifact_path.txt')
    artifact_path = last_used_model_obj['Body'].read().decode('utf-8').strip()
    print(f"Artifact path retrieved: {artifact_path}")
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        s3.download_fileobj(bucket_name, artifact_path, temp_file)
        temp_file_path = temp_file.name

    model = joblib.load(temp_file_path)
    os.remove(temp_file_path)
    print("Model loaded successfully.")

    return model


def preprocess_input(request):
    df = pd.DataFrame([request.dict()])

    df = df.dropna()

    gender_mapping = {"Male": 0, "Female": 1}
    df["gender"] = df["gender"].map(gender_mapping)

    boolean_columns = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in boolean_columns:
        df[col] = df[col].astype(int)

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

    return df
