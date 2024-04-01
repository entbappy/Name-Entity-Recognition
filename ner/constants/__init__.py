import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
LOGS_DIR = "logs"
LOGS_FILE_NAME = "ner.log"
MODELS_DIR = "models"
BEST_MODEL_DIR = "best_model"

BUCKET_NAME = "ner-using-bert-24"
GCP_DATA_FILE_NAME = "archive.zip"
CSV_DATA_FILE_NAME = "ner.csv"
GCP_MODEL_NAME = "model.pt"

DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"