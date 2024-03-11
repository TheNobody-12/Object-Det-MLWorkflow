import os
from dataclasses import dataclass
from datetime import datetime
from src.HelmetDetection.constants.training_pipeline import *

@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = ARTIFACTS_DIR

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION
    )

    feature_store_dir: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR
    )

    data_download_url: str = DATA_DOWNLOAD_URL


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_VALIDATION_DIR_NAME
    )

    valid_status_file_dir:str = os.path.join(
        data_validation_dir, DATA_VALIDATION_STATUS_FILE
    )

    required_file_list = DATA_VALIDATION_ALL_REQUIRED_FILES


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME
    )

    pretrained_weights: str = MODEL_TRAINER_PRETRAINED_WEIGHTS

    no_epochs: int = MODEL_TRAINER_NO_EPOCHS

    batch_size: int = MODEL_TRAINER_BATCH_SIZE

    img_size: int = MODEL_TRAINER_IMG_SIZE

    device: int = MODEL_TRAINER_DEVICE

    workers: int = MODEL_TRAINER_WORKERS

    project: str = MODEL_TRAINER_PROJECT

    name: str = MODEL_TRAINER_NAME

    nms: bool = MODEL_TRAINER_NMS

    fraction: float = MODEL_TRAINER_FRACTION

    format: str = MODEL_TRAINER_FORMAT

    