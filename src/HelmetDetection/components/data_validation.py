import os
import sys
import zipfile
import gdown
from ultralytics.data.converter import convert_coco
import yaml
import shutil
from src.HelmetDetection.logger import logging
from src.HelmetDetection.entity.config_entity import DataValidationConfig
from src.HelmetDetection.entity.artifacts_entity import (
    DataIngestionArtifacts, DataValidationArtifacts)
from src.HelmetDetection.exception import AppException


class DataValidation:
    def __init__(self, data_ingestion_artifacts: DataIngestionArtifacts, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise AppException(e, sys)

    def validate_all_files_exit(self) -> bool:
        try:
            validation_status = None

            dir_path = self.data_ingestion_artifacts.feature_store_path + "/yolo_data"

            all_files = os.listdir(dir_path)

            for file in all_files:
                if file not in self.data_validation_config.required_file_list:
                    validation_status = False
                    os.makedirs(
                        self.data_validation_config.data_validation_dir, exist_ok=True)
                    with open(self.data_validation_config.valid_status_file_dir, "w") as file:
                        file.write(f"Validation Status: {validation_status}")

                else:
                    validation_status = True
                    os.makedirs(
                        self.data_validation_config.data_validation_dir, exist_ok=True)
                    with open(self.data_validation_config.valid_status_file_dir, "w") as file:
                        file.write(f"Validation Status: {validation_status}")

            return validation_status

        except Exception as e:
            raise AppException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifacts:
        logging.info("Initiating Data Validation")
        try:
            validation_status = self.validate_all_files_exit()
            data_validation_artifacts = DataValidationArtifacts(
                validation_status=validation_status)
            logging.info("Data Validation Completed")
            logging.info(f"Data Validation Status: {validation_status}")
            if validation_status:
                shutil.copy(self.data_ingestion_artifacts.data_zip_file_path,
                            os.getcwd())
            return data_validation_artifacts
        except Exception as e:
            raise AppException(e, sys)
