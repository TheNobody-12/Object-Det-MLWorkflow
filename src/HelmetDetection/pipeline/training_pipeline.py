import sys
import os
from src.HelmetDetection.logger import logging
from src.HelmetDetection.exception import AppException
from src.HelmetDetection.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, ModelTrainerConfig)
from src.HelmetDetection.components.data_ingestion import (DataIngestion)
from src.HelmetDetection.components.data_validation import (DataValidation)
from src.HelmetDetection.components.model_trainer import (ModelTrainer)
from src.HelmetDetection.entity.artifacts_entity import (
    DataIngestionArtifacts, DataValidationArtifacts, ModelTrainerArtifacts)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            logging.info("Starting data ingestion process")
            data_ingestion = DataIngestion(self.data_ingestion_config)

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion process completed")
            logging.info(
                f"Data ingestion artifacts: {data_ingestion_artifacts}")
            return data_ingestion_artifacts
        except Exception as e:
            raise AppException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifacts) -> DataValidationArtifacts:
        logging.info("Starting data validation process")
        try:
            data_validation = DataValidation(
                data_ingestion_artifacts=data_ingestion_artifact, data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation process completed")
            logging.info(
                f"Data validation artifacts: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise AppException(e, sys)

    def start_Model_training(self, data_validation_artifact: DataValidationArtifacts) -> ModelTrainerArtifacts:
        try:
            model_trainer_config = ModelTrainerConfig()
            logging.info("Starting model training process")
            model_trainer = ModelTrainer(
                data_validation_artifact=data_validation_artifact, model_trainer_config=model_trainer_config)
            model_trainer_artifacts = model_trainer.run_model_training()
            logging.info("Model training process completed")
            logging.info(
                f"Model training artifacts: {model_trainer_artifacts}")
            return model_trainer_artifacts
        except Exception as e:
            raise AppException(e, sys)
        

    def run_pipeline(self) -> None:
        STAGE_NAME = "Data Ingestion stage"
        try:
            logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            data_ingestion_artifacts = self.start_data_ingestion()
            # data_ingestion_artifacts = "artifacts\data_ingestion\\feature_store"
            logging.info("Data ingestion process completed")
            logging.info(
                f"Data ingestion artifacts: {data_ingestion_artifacts}")
            logging.info(
                f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")

            STAGE_NAME = "Data Validation stage"
            logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            data_validation_artifacts = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifacts)
            logging.info("Data validation process completed")
            logging.info(
                f"Data validation artifacts: {data_validation_artifacts}")
            logging.info( f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
            # data_validation_artifacts = True

            STAGE_NAME = "Model Training stage"
            logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            model_trainer_artifacts = self.start_Model_training(
                data_validation_artifact=data_validation_artifacts)
            logging.info("Model training process completed")
            logging.info(
                f"Model training artifacts: {model_trainer_artifacts}")
            logging.info(
                f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
            
            logging.info(f"Training pipeline completed successfully")


        except Exception as e:
            raise AppException(e, sys)
