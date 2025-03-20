import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml
from config.paths_config import *

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.bucket_name = config["data_ingestion"]["bucket_name"]
        self.file_name = config["data_ingestion"]["bucket_file_name"]
        self.train_test_ratio = config["data_ingestion"]["train_ratio"]
        
        os.makedirs(RAW_DIR,exist_ok=True)

        logger.info("Data Ingestion object created {self.bucket_name} and file save is {self.file_name}")

    def download_data_from_bucket(self):
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Data downloaded from bucket {self.bucket_name} and file save is {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error occured while downloading the data from bucket {e}")
            raise CustomException("Error occured while downloading the data from bucket",e)
        
    def split_data(self):
        try:
            logger.info("Starting to split the data")
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data, test_size=1-self.train_test_ratio, random_state=42)
            logger.info("Data split completed")

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Train and Test data saved at {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error occured while splitting the data {e}")
            raise CustomException("Error occured while splitting the data",e)
        
    def run (self):
        self.download_data_from_bucket()
        self.split_data()
        logger.info("Data Ingestion run completed")

if __name__ == "__main__":
    run_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    run_ingestion.run()