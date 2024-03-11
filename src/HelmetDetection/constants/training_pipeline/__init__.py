ARTIFACTS_DIR: str = "artifacts"

DATA_INGESTION: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1OT8qzsV-aiMxdEEgIkjQRm1HUR6cI0Zu/view?usp=sharing"

"""
Data Validation related constants start with Data Validation
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATUS_FILE = "status.txt"

DATA_VALIDATION_ALL_REQUIRED_FILES = ["images", "labels", "data.yaml"]


"""
Model Trainer related constants start with Model Trainer
model,epochs,data,imgsz,device,batch,coslr,workers,project,name,nms,fraction,format
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_PRETRAINED_WEIGHTS: str = "yolov8n.pt"

MODEL_TRAINER_NO_EPOCHS: int = 1

MODEL_TRAINER_BATCH_SIZE: int = 1

MODEL_TRAINER_IMG_SIZE: int = 640

MODEL_TRAINER_DEVICE: int = 0

MODEL_TRAINER_WORKERS: int = 10

MODEL_TRAINER_PROJECT: str = "HelmetDet"

MODEL_TRAINER_NAME: str = "yolov8s"

MODEL_TRAINER_NMS: bool = True

MODEL_TRAINER_FRACTION: float = 1.0

MODEL_TRAINER_FORMAT: str = "onnx"









