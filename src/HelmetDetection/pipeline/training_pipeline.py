import sys
import os
from src.HelmetDetection.logger import logging
from src.HelmetDetection.exception import AppException
from src.HelmetDetection.entity.config_entity import (DataIngestionConfig)
from src.HelmetDetection.components.data_ingestion import (DataIngestion)
from src.HelmetDetection.entity.artifacts_entity import (
    DataIngestionArtifacts)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

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
            raise AppException(e, sys) from e

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            logging.info("Data ingestion process completed")
            logging.info(
                f"Data ingestion artifacts: {data_ingestion_artifacts}")
        except Exception as e:
            raise AppException(e, sys) from e
