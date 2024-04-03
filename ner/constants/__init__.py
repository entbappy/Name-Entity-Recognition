import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
LOGS_DIR = "logs"
LOGS_FILE_NAME = "ner.log"
MODELS_DIR = "models"
BEST_MODEL_DIR = "best_model"

# data ingestion 
BUCKET_NAME = "ner-using-bert-24"
GCP_DATA_FILE_NAME = "archive.zip"
CSV_DATA_FILE_NAME = "ner.csv"
GCP_MODEL_NAME = "model.pt"

DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"


# data transformations

DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
LABELS_TO_IDS_FILE_NAME = "labels_to_ids.pkl"
IDS_TO_LABELS_FILE_NAME = "ids_to_labels.pkl"
DF_TRAIN_FILE_NAME = "df_train.pkl"
DF_VAL_FILE_NAME = "df_val.pkl"
DF_TEST_FILE_NAME = "df_test.pkl"
UNIQUE_LABELS_FILE_NAME = "unique_labels.pkl"