
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *

if __name__ == "__main__":
    ### Data Ingestion
    run_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    run_ingestion.run()

    ### Data Processing
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()  

    ### Model Training
    trainer = ModelTraining(train_path=PROCESSED_TRAIN_FILE_PATH,
                            test_path=PROCESSED_TEST_FILE_PATH,
                            model_output_path=MODEL_OUTPUT_PATH)
    trainer.run()