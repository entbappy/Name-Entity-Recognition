import os
import sys
from zipfile import ZipFile
from ner.configuration.gcloud import GCloud
from ner.constants import *
from ner.entity.artifact_entity import DataIngestionArtifacts
from ner.entity.config_entity import DataIngestionConfig
from ner.exception import NerException
from ner.logger import logging


class DataIngestion:
    def __init__(
        self, data_ingestion_config: DataIngestionConfig, gcloud: GCloud
    ) -> None:
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = gcloud


    def get_data_from_gcp(self, bucket_name: str, file_name: str, path: str) -> ZipFile:
        logging.info("Entered the get_data_from_gcp method of data ingestion class")
        try:
            self.gcloud.sync_folder_from_gcloud(
                gcp_bucket_url=bucket_name, filename=file_name, destination=path
            )
            logging.info("Exited the get_data_from_gcp method of data ingestion class")

        except Exception as e:
            raise NerException(e, sys) from e


    def extract_data(self, input_file_path: str, output_file_path: str) -> None:
        logging.info("Entered the extract_data method of Data ingestion class")
        try:
            # loading the temp.zip and creating a zip object
            with ZipFile(input_file_path, "r") as zObject:

                # Extracting all the members of the zip
                # into a specific location.
                zObject.extractall(path=output_file_path)
            logging.info("Exited the extract_data method of Data ingestion class")

        except Exception as e:
            raise NerException(e, sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info(
            "Entered the initiate_data_ingestion method of data ingestion class"
        )
        try:
            # Creating Data Ingestion Artifacts directory inside artifacts folder
            os.makedirs(
                self.data_ingestion_config.data_ingestion_artifacts_dir, exist_ok=True
            )
            logging.info(
                f"Created {os.path.basename(self.data_ingestion_config.data_ingestion_artifacts_dir)} directory."
            )

            # Getting data from GCP
            self.get_data_from_gcp(
                bucket_name=BUCKET_NAME,
                file_name=GCP_DATA_FILE_NAME,
                path=self.data_ingestion_config.gcp_data_file_path,
            )
            logging.info(
                f"Got the file from Google cloud storage. File name - {os.path.basename(self.data_ingestion_config.gcp_data_file_path)}"
            )

            # Extracting the data file
            self.extract_data(
                input_file_path=self.data_ingestion_config.gcp_data_file_path,
                output_file_path=self.data_ingestion_config.output_file_path,
            )
            logging.info(f"Extracted the data from zip file.")

            data_ingestion_artifact = DataIngestionArtifacts(
                zip_data_file_path=self.data_ingestion_config.gcp_data_file_path,
                csv_data_file_path=self.data_ingestion_config.csv_data_file_path,
            )
            logging.info("Exited the initiate_data_ingestion method of data ingestion class")
            return data_ingestion_artifact

        except Exception as e:
            raise NerException(e, sys) from e
