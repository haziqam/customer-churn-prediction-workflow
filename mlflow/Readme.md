# MLflow Workflow for Customer Churn Prediction

This project implements an end-to-end machine learning workflow for customer churn prediction using **MLflow**. It includes data preprocessing, model training, drift detection, and retraining. The project is designed to integrate seamlessly with tools like **Apache Airflow** for orchestration and **Apache Spark** for scalable data processing.

---

## Project Structure

```
mlflow_project/
├── Dockerfile                # Docker configuration for MLflow server
├── docker-compose.yml        # Orchestrates MLflow services with PostgreSQL and MinIO
├── requirements.txt          # Python dependencies for MLflow client
├── scripts/                  # Scripts for MLflow workflows
│   ├── preprocess.py         # Data preprocessing and logging
│   ├── train_model.py        # Model training and evaluation
│   ├── drift_detection.py    # PSI calculation for drift detection
│   ├── retrain_model.py      # Retraining workflow
│   └── utils.py              # Shared utility functions
├── models/                   # Folder for storing local model files
│   └── random_forest_model.pkl
├── data/                     # Folder for raw and preprocessed datasets
│   ├── raw_data.csv          # Input raw data
│   └── processed_data.csv    # Cleaned and preprocessed data
├── artifacts/                # Local storage for MLflow artifacts
├── postgres_data/            # Persistent storage for PostgreSQL
├── minio_data/               # Persistent storage for MinIO
└── README.md                 # Documentation (this file)
```

## Workflow Overview
This MLflow workflow consists of four key steps:

1. Data Preprocessing
- Script: scripts/preprocess.py
- Purpose: Cleans the raw dataset, handles missing values, and saves the preprocessed data.
- Logs in MLflo
- Parameters: Missing value strategy, initial data shape.
- Metrics: Number of missing values handled.
- Artifacts: Preprocessed data.
- Input: data/raw_data.csv
- Output: data/processed_data.csv

2. Model Training
- Script: scripts/train_model.py
- Purpose: Trains a Random Forest model, evaluates it, and logs all details to MLflow.
- Logs in MLflow:
- Parameters: Model type, number of estimators.
- Metrics: Accuracy, F1 score.
- Artifacts: Trained model.
- Input: data/processed_data.csv
- Output: models/random_forest_model.pkl

3. Drift Detection
- Script: scripts/drift_detection.py
- Purpose: Calculates the Population Stability Index (PSI) to detect data drift and logs drift metrics to MLflow.
- Logs in MLflow:
- Metrics: PSI value.
- Parameters: PSI threshold, drift detection flag.
- Input: Training and production distributions.
- Output: Drift metrics logged in MLflow.

4. Retraining Workflow
- Script: scripts/retrain_model.py
- Purpose: Automatically retrains the model if data drift is - detected, ensuring the model remains accurate.
- Logs in MLflow:
- Retrained model parameters and metrics.
- Updated model artifact.
- Input: data/processed_data.csv (new data).
- Output: models/retrained_model.pkl

## Integration with Other Tools
1. Apache Airflow
Apache Airflow orchestrates and schedules the entire ML workflow. Example DAGs automate preprocessing, training, drift detection, and retraining.
Workflow in Airflow:
```bash
Preprocessing → Training → Drift Detection → Conditional Retraining
```

2. Apache Spark
Apache Spark can handle data preprocessing and feature engineering for large datasets before feeding data into MLflow workflows.

3. Dockerized Setup
The project runs inside Docker containers for portability and reproducibility. Docker Compose orchestrates MLflow, PostgreSQL, and MinIO.

Services:
- MLflow Tracking Server: Tracks experiments and manages the model registry.
- PostgreSQL: Stores metadata (e.g., run details, metrics).
- MinIO: Acts as the artifact storage (similar to AWS S3).

## How to Run
1. Step 1: Clone, Build, and Start MLflow Server
```bash
git clone https://github.com/your-repo/mlflow-project.git
cd mlflow-project
```
Build and start Docker containers:
```bash
docker-compose up -d --build
```
Access MLflow Tracking Server:
```
URL: http://localhost:5000
MinIO Console: http://localhost:9001 (credentials: minio / minio123)
```
2. Step 2: Run Scripts
Preprocess Data:
```bash
python scripts/preprocess.py
```

Train Model:
```bash
python scripts/train_model.py
```

Run Drift Detection:
```bash
python scripts/drift_detection.py
```
Retrain Model (if drift detected):
```bash
python scripts/retrain_model.py
```
3. Step 3: Integrate with Airflow
Add Airflow DAGs for each workflow step (mlflow_workflow.py).
Schedule DAGs to run automatically.
Monitor execution in the Airflow UI (http://localhost:8080).
