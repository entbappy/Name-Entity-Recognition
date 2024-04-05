import sys
from ner.components.data_ingestion import DataIngestion
from ner.components.data_transforamation import DataTransformation
from ner.components.model_trainer import ModelTraining
from ner.components.model_evaluation import ModelEvaluation
from ner.components.model_pusher import ModelPusher
from ner.configuration.gcloud import GCloud
from ner.constants import *

from ner.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainingArtifacts,
    ModelEvaluationArtifacts,
    ModelPusherArtifacts
    )


from ner.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvalConfig,
    ModelPusherConfig
    
)


from ner.exception import NerException
from ner.logger import logging


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_training_config = ModelTrainingConfig()
        self.model_evaluation_config = ModelEvalConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.gcloud = GCloud()

    
     # This method is used to start the data ingestion
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from Google cloud storage")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config, gcloud=self.gcloud
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from Google cloud storage")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact

        except Exception as e:
            raise NerException(e, sys) from e
        


    
     # This method is used to start the data validation
    def start_data_transformation(
        self, data_ingestion_artifact: DataIngestionArtifacts
    ) -> DataTransformationArtifacts:
        logging.info(
            "Entered the start_data_transformation method of TrainPipeline class"
        )
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifacts=data_ingestion_artifact,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logging.info("Performed the data validation operation")
            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )
            return data_transformation_artifact

        except Exception as e:
            raise NerException(e, sys) from e
        

    
    # This method is used to start the model trainer
    def start_model_training(
        self, data_transformation_artifacts: DataTransformationArtifacts
    ) -> ModelTrainingArtifacts:
        logging.info("Entered the start_model_training method of Train pipeline class")
        try:
            model_trainer = ModelTraining(
                model_trainer_config=self.model_training_config,
                data_transformation_artifacts=data_transformation_artifacts,
            )
            model_trainer_artifact = model_trainer.initiate_model_training()

            logging.info("Performed the Model training operation")
            logging.info(
                "Exited the start_model_training method of Train pipeline class"
            )
            return model_trainer_artifact

        except Exception as e:
            raise NerException(e, sys) from e
        

    

     # This method is used to start model evaluation
    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifacts,
        model_trainer_artifact: ModelTrainingArtifacts,
    ) -> ModelEvaluationArtifacts:
        try:
            logging.info(
                "Entered the start_model_evaluation method of Train pipeline class"
            )
            model_evaluation = ModelEvaluation(
                data_transformation_artifacts=data_transformation_artifact,
                model_training_artifacts=model_trainer_artifact,
                model_evaluation_config=self.model_evaluation_config,
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            logging.info(
                "Exited the start_model_evaluation method of Train pipeline class"
            )
            return model_evaluation_artifact

        except Exception as e:
            raise NerException(e, sys) from e
        

    


     # This method is used to statr model pusher
    def start_model_pusher(
        self, model_evaluation_artifact: ModelEvaluationArtifacts
    ) -> ModelPusherArtifacts:
        try:
            logging.info(
                "Entered the start_model_pusher method of Train pipeline class"
            )
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            logging.info("Exited the start_model_pusher method of Train pipeline class")
            return model_pusher_artifact

        except Exception as e:
            raise NerException(e, sys) from e

        


    

      # This method is used to start the training pipeline
    def run_pipeline(self) -> None:
        try:
            logging.info("Started Model training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            model_trainer_artifact = self.start_model_training(
                data_transformation_artifacts=data_transformation_artifacts
            )
            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact=data_transformation_artifacts,
                model_trainer_artifact=model_trainer_artifact,
            )

            model_pusher_artifact = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )

        except Exception as e:
            raise NerException(e, sys) from e
