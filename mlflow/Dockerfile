# Use official Python image
FROM python:3.9-slim

# Set environment variables
ENV MLFLOW_HOME=/mlflow
ENV PATH=$MLFLOW_HOME/bin:$PATH
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    gcc \
    && apt-get clean

# Create a directory for MLflow
WORKDIR $MLFLOW_HOME

COPY scripts /mlflow/scripts

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for MLflow Tracking Server
EXPOSE 5000

# Run the MLflow Tracking Server
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@db:5432/mlflow", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0"]
