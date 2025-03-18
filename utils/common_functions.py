import os 
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info("Successfully raed the yaml file")
                return config
        else:
            raise FileNotFoundError(f"File not found at path {file_path}",sys.exc_info())
    except Exception as e:
        logger.error(f"Error occured while reading the yaml file {e}")
        raise CustomException("Error occured while reading the yaml file",e)