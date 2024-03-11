import os
import sys
import yaml
from src.HelmetDetection.logger import logging
from src.HelmetDetection.exception import AppException
from src.HelmetDetection.entity.config_entity import ModelTrainerConfig, DataIngestionConfig, DataValidationConfig
from src.HelmetDetection.entity.artifacts_entity import ModelTrainerArtifacts, DataIngestionArtifacts, DataValidationArtifacts
import mlflow

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from ultralytics import settings
from ultralytics import YOLO
import torch.nn as nn
import torch
import argparse
import re

settings.update({'mlflow': False,
                 'datasets_dir': 'F:\IMP_DOCUMENT\Projects\DEVOPs\ObjectBB\\artifacts\data_ingestion\\feature_store',
                 'weights_dir': "artifacts\models",
                 'runs_dir': "artifacts\models"})

# Define a wrapper class to ensure pytorch_model is an instance of torch.nn.Module


class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
# mlflow.autolog()


def on_train_end(trainer):
    #    if mlflow:
    print('in the on_train_end callbacks')
    print('trainers with trainers.metrics:' + str(trainer.best))
    print('trainers with trainers.metrics:' + str(trainer.last))
    print('trainers with trainers.metrics:' + str(trainer.testset))
    print('trainers with trainers.metrics:' + str(trainer.trainset))
    mlflow.log_artifact(str(trainer.best), "model")

#    mlflow.pytorch.log_model(str(trainer.best), "model_log")
    # End the MLflow run
    # # Convert to ONNX first
    # model.export(format='onnx', path='model.onnx')
    #
    # # Then log the ONNX model to MLflow
    # mlflow.log_artifact('model.onnx', artifact_path="models")

    # Register the model in MLflow Model Registry
    # mlflow.register_model(
    #      "runs:/train/mlflow_simple_new/model",
    #      "my_registered_model"
    #  )

    # # Register the model in MLflow Model Registry
    # mlflow.register_model(
    #     "runs:/my_project/my_experiment/model",
    #     "my_registered_model"
    # )


def on_train_start(trainer):
    #    if mlflow:
    #    print('in the on_train_start callbacks trainer'+trainer)
    print('trainers with trainers.metrics:' + str(trainer.best))
    print('trainers with trainers.metrics:' + str(trainer.last))
    print('trainers with trainers.metrics:' + str(trainer.testset))
    print('trainers with trainers.metrics:' + str(trainer.trainset))


def on_fit_epoch_end(trainers):
    metrics_dict = {f"{re.sub('[()]', '', k)}": float(v)
                    for k, v in trainers.metrics.items()}
    print('trainers.metrics with metrics_dict:' + str(metrics_dict))
    mlflow.log_metrics(metrics=metrics_dict, step=trainers.epoch)


def convert_pt_to_pytorch_model(model_path):
    # Load the .pt model
    model = torch.load(model_path)
    # Convert it to a PyTorch model if necessary
    # For example:
    # pytorch_model = YourYOLOModel(your_model_arguments)
    # Copy the parameters from the loaded model to the PyTorch model
    # pytorch_model.load_state_dict(model.state_dict())
    return model

# create a class to train the model


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_validation_artifact: DataValidationArtifacts):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise AppException(e, sys) from e

    def start_model_training(self,) -> ModelTrainerArtifacts:
        logging.info("Starting model training process")
        try:
            # Create a YOLO model
            model = YOLO(self.model_trainer_config.pretrained_weights)
            model.add_callback('on_train_end', on_train_end)
            model.add_callback('on_train_start', on_train_start)
            model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

            # Train the model
            results = model.train(
                task="detect",
                data="data.yaml",
                epochs=self.model_trainer_config.no_epochs,
                batch=self.model_trainer_config.batch_size,
                imgsz=self.model_trainer_config.img_size,
                # device=self.model_trainer_config.device,
                workers=self.model_trainer_config.workers,
                project=self.model_trainer_config.project,
                name=self.model_trainer_config.name,
                nms=self.model_trainer_config.nms,
                fraction=self.model_trainer_config.fraction,
                format=self.model_trainer_config.format
            )

            # Log the metrics
            mlflow.log_metrics({
                "Precision": results.results_dict['metrics/precision(B)'],
                "Recall": results.results_dict['metrics/recall(B)'],
                "map": results.results_dict['metrics/mAP50-95(B)'],
                "map50": results.results_dict['metrics/mAP50(B)'],
            })

            # log model
            if self.model_trainer_config.format == "pytorch":
                pymodel = convert_pt_to_pytorch_model(results.best)
                wrapped_model = WrapperModel(pymodel)
                # Save the wrapped_model with MLflow
                mlflow.pytorch.log_model(wrapped_model, artifact_path="pytorch_model",
                                         registered_model_name=self.model_trainer_config.name)
                mlflow.pytorch.save_model(
                    wrapped_model, path=f"{self.model_trainer_config.project}/{self.model_trainer_config.name}/pytorch_model", pip_requirements="requirements.txt")
                model_path = f"{self.model_trainer_config.project}/{self.model_trainer_config.name}/pytorch_model"
            elif self.model_trainer_config.format == "onnx":
                model.export(format='onnx')
                mlflow.log_artifact(
                    f'{self.model_trainer_config.project}/{self.model_trainer_config.name}/weights/best.onnx', artifact_path="artifacts/models")
                mlflow.register_model(
                    f'{self.model_trainer_config.project}/{self.model_trainer_config.name}/weights/best.onnx', "my_registered_model")
                model_path = f'{self.model_trainer_config.project}/{self.model_trainer_config.name}/weights/best.onnx'
            return ModelTrainerArtifacts(
                trained_model_path=model_path,
            )
        except Exception as e:
            raise AppException(e, sys) from e

    def run_model_training(self) -> None:
        try:
            if self.data_validation_artifact:         
                mlflow.set_experiment(self.model_trainer_config.project)
                with mlflow.start_run(run_name=self.model_trainer_config.name):
                    mlflow.log_params(vars(self.model_trainer_config))
                    model_training_artifacts = self.start_model_training()
                    logging.info("Model training process completed")
                    logging.info(
                        f"Model training artifacts: {model_training_artifacts}")
                    
                    mlflow.log_artifacts(
                        self.model_trainer_config.project, artifact_path="artifacts")
                    mlflow.end_run()
                    return model_training_artifacts 
            else:
                logging.info("Data validation process failed")
                logging.info(
                    f"Data validation artifacts: {self.data_validation_artifact}")
        except Exception as e:
            raise AppException(e, sys)
