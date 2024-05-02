import os
import sys
import zipfile
import gdown
from ultralytics.data.converter import convert_coco
import yaml
import shutil
from src.HelmetDetection.logger import logging
from src.HelmetDetection.entity.config_entity import DataIngestionConfig
from src.HelmetDetection.entity.artifacts_entity import DataIngestionArtifacts
from src.HelmetDetection.exception import AppException


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise AppException(e, sys) from e
        

    def download_data(self) -> str:
        '''fetch data from url'''
        try:
            data_url = self.data_ingestion_config.data_download_url
            zip_download_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_download_dir, exist_ok=True)
            data_file_name = "data.zip"
            zip_file_path = os.path.join(zip_download_dir, data_file_name)
            logging.info(f"Downloading data from {data_url} into file {zip_file_path}")

            file_id = data_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?id="
            url = prefix + file_id
            gdown.download(url, zip_file_path, quiet=False)
            logging.info(f"Downloaded data from {data_url} into file {zip_file_path}")
            
            return zip_file_path
        except Exception as e:
            raise AppException(e, sys) from e
        

    def extract_data(self, zip_file_path: str) -> str:
        '''extract data from zip file'''
        try:
            feature_store_dir = self.data_ingestion_config.feature_store_dir
            os.makedirs(feature_store_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(feature_store_dir)
            logging.info(f"Conversion  of COCO to YOLO format")
            # create annotation dir if not exists
            if not os.path.exists(f"{feature_store_dir}/annotations"):
                os.makedirs(f"{feature_store_dir}/annotations")


            # move annotations to annotations dir
            shutil.move(f"{feature_store_dir}/COCO/annotations/instances_Images.json", f"{feature_store_dir}/annotations/train.json")
            shutil.move(f"{feature_store_dir}/COCO/annotations/instances_val2014.json", f"{feature_store_dir}/annotations/test.json")

            # convert COCO to YOLO format
            convert_coco(f"{feature_store_dir}/annotations", f"{feature_store_dir}/yolo_data")

            # copy images to yolo_data
            shutil.copytree(f"{feature_store_dir}/COCO/train2014", f"{feature_store_dir}/yolo_data/images/train")
            shutil.copytree(f"{feature_store_dir}/COCO/val2014", f"{feature_store_dir}/yolo_data/images/test")
            
            # the data.yaml file is used by the train.py file
            data_yaml = dict(
                train = '../yolo_data/images/train',
                val = '../yolo_data/images/test',
                nc = 9,
                names = ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet', 'P0Helmet', 'P0NoHelmet']
            )
            logging.info(f"Writing data.yaml file")
            logging.info(f"successfully converted COCO to YOLO format")

            # write the data.yaml file
            with open(f'{feature_store_dir}/yolo_data/data.yaml', 'w') as outfile:
                yaml.dump(data_yaml, outfile, default_flow_style=False)

            logging.info(f"Extracted data from {zip_file_path} into directory {feature_store_dir}")
            return feature_store_dir
        except Exception as e:
            raise AppException (e, sys) from e
        

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        '''initiate data ingestion process'''
        try:
            zip_file_path = self.download_data()
            # zip_file_path = "artifacts/data_ingestion/data.zip"
            feature_store_dir = self.extract_data(zip_file_path)
            data_ingestion_artifacts = DataIngestionArtifacts(
                data_zip_file_path=zip_file_path,
                feature_store_path=feature_store_dir
                
            )
            logging.info(f"Data ingestion process completed successfully")
            return data_ingestion_artifacts
        except Exception as e:
            raise AppException(e, sys) from e
        
