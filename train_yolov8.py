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
                 "datasets_dir":"D:\Sarthak\Object-Det-MLWorkflow"})

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
    #mlflow.log_artifact('model.onnx', artifact_path="models")

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
    print('trainers with trainers.metrics:'+ str(trainer.best))
    print('trainers with trainers.metrics:'+ str(trainer.last))
    print('trainers with trainers.metrics:'+ str(trainer.testset))
    print('trainers with trainers.metrics:'+ str(trainer.trainset))

def on_fit_epoch_end(trainers):
    metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainers.metrics.items()}
    print('trainers.metrics with metrics_dict:'+ str(metrics_dict))
    mlflow.log_metrics(metrics=metrics_dict, step=trainers.epoch)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a YOLO model")
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--data", type=str, default="YOLO_DATA/data.yaml", help="path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--device", type=int, default=0, help="device id (i.e. 0 or 0,1 or cpu)")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--cos_lr", action="store_true", help="use cosine learning rate")
    parser.add_argument("--workers", type=int, default=10, help="number of workers")
    parser.add_argument("--optimizer", type=str, default="auto", help="optimizer")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--project", type=str, default="HelmetDet", help="project name")
    parser.add_argument("--name", type=str, default="yolov8n", help="run name")
    parser.add_argument("--nms", action="store_true", help="use non-maximum suppression")
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of dataset to use")
    parser.add_argument("--format", type=str, default="pytorch", help="model format")
    args = parser.parse_args()
    return args

def convert_pt_to_pytorch_model(model_path):
    # Load the .pt model
    model = torch.load(model_path)
    # Convert it to a PyTorch model if necessary
    # For example:
    # pytorch_model = YourYOLOModel(your_model_arguments)
    # Copy the parameters from the loaded model to the PyTorch model
    # pytorch_model.load_state_dict(model.state_dict())
    return model

if __name__ == '__main__':
    # Start an MLflow run
    args = parse_args()
    mlflow.set_experiment(args.project)
    with mlflow.start_run(run_name=args.name):
        # Parse the arguments
        # Log all of the arguments
        mlflow.log_params(vars(args))
        
        # Your training code here
        model = YOLO(args.model)
        model.add_callback("on_train_start",on_train_start)
        model.add_callback("on_fit_epoch_end",on_fit_epoch_end)
        model.add_callback("on_train_end", on_train_end)
        results = model.train(task="detect",data=args.data, imgsz=args.imgsz, batch=args.batch, epochs=args.epochs, device=args.device, project=args.project, nms=args.nms,cos_lr=args.cos_lr, optimizer=args.optimizer, seed=args.seed, name=args.name, resume=args.resume, fraction=args.fraction)
        
        # Log the metrics
        mlflow.log_metrics({
            "Precision":results.results_dict['metrics/precision(B)'],
            "Recall":results.results_dict['metrics/recall(B)'],
            "map":results.results_dict['metrics/mAP50-95(B)'],
            "map50":results.results_dict['metrics/mAP50(B)'],
        })
        
        # Log artifacts of project to MLflow

        if args.format == 'onnx':
            # Convert to ONNX first
            model.export(format='onnx')
            # load the model
            # model = mlflow.onnx.load_model("model_log")
            # mlflow.onnx.log_model(model, artifact_path="models")
            # Then log the ONNX model to MLflow
            mlflow.log_artifact(f'{args.project}/{args.name}/weights/best.onnx', artifact_path="models")
            mlflow.register_model(
                f'{args.project}/{args.name}/weights/best.onnx',
                "my_registered_model"
            )

        elif args.format == 'pytorch':
            # Convert the .pt model to PyTorch model
            pytorch_model = convert_pt_to_pytorch_model(f"{args.project}/{args.name}/weights/best.pt")
            # Log the PyTorch model with MLflow
            # Wrap your pytorch_model
            wrapped_model = WrapperModel(pytorch_model)
            # Save the wrapped_model with MLflow
            mlflow.pytorch.log_model(wrapped_model, artifact_path="pytorch_model",registered_model_name=args.name)
            mlflow.pytorch.save_model(wrapped_model, path=f"{args.project}/{args.name}/pytorch_model",pip_requirements="requirements.txt")
        
        mlflow.log_artifacts(f"{args.project}/{args.name}")
        # End the MLflow run
        mlflow.end_run()