import os.path
import sys
import yaml
import base64

from src.HelmetDetection.logger import logging
from src.HelmetDetection.exception import AppException

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)
    except Exception as e:
        # logging.error(e)
        raise AppException(e, sys) from e
    

def write_yaml_file(file_path:str, content:object, replace:bool=False):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path) 
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            yaml.dump(content, yaml_file, default_flow_style=False)
            logging.info("Write yaml file successfully")
    except Exception as e:
        # logging.error(e)
        raise AppException(e, sys) from e
   
def decodeImage(imgstring,fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()
    logging.info("Decode image successfully")

def encodeImage(croppedImagePath):
    with open(croppedImagePath,"rb") as f:
        return base64.b64encode(f.read())
    
 