from src.HelmetDetection.logger import logging
from src.HelmetDetection.exception import AppException
import sys

try:
    a = 3/ "small"
except Exception as e:
    raise AppException(e, sys)