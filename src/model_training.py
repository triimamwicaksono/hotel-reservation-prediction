
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEACH_PARAMS


    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data splitted sucefully for Model Training")

            return X_train,y_train,X_test,y_test
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data" ,  e)

    def train_lgbm(self,X_train,y_train):
        try:
            logger.info("Intializing our model")

            lgbm_model = lgb.LGBMClassifier(random_state=42)

            logger.info("Starting our Hyperparamter tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter = self.random_search_params["n_iter"],
                cv = self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )

            logger.info("Starting our Hyperparamter tuning")

            random_search.fit(X_train,y_train)

            logger.info("Hyperparamter tuning completed")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best paramters are : {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model" ,  e)
            
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Staring Model Evaluation")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            logger.info(f"Accuracy : {accuracy}")
            logger.info(f"Precision : {precision}")    
            logger.info(f"Recall : {recall}")
            logger.info(f"F1 Score : {f1}")
            return {
                "accuracy":accuracy,
                "precision":precision,
                "recall":recall,
                "f1_score":f1
            }

        except Exception as e:
            logger.error(f"Error occured while evaluating the model {e}")
            raise CustomException("Model Evaluation Failed",e)

    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            logger.info("Saving the model")
            joblib.dump(model,self.model_output_path)
            logger.info(f"Model saved at {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error occured while saving the model {e}")
            raise CustomException("Model Saving Failed",e)
    
    def run(self):
        try:
            with mlflow.start_run():
                mlflow.log_artifact(self.train_path, artifact_path = "datasets")
                mlflow.log_artifact(self.test_path, artifact_path = "datasets")
                logger.info("Staring our model training pipeline")
                X_train,y_train,X_test,y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train,y_train)
                metrics = self.evaluate_model(best_lgbm_model,X_test,y_test)
                self.save_model(best_lgbm_model)
                mlflow.log_artifact(self.model_output_path, artifact_path = "models")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)
                logger.info("Model Training Pipeline Completed")
        except Exception as e:  
            logger.error(f"Error occured while running the model training pipeline {e}")
            raise CustomException("Model Training Pipeline Failed",e)
        
if __name__ == "__main__":

    trainer = ModelTraining(train_path=PROCESSED_TRAIN_FILE_PATH,
                            test_path=PROCESSED_TEST_FILE_PATH,
                            model_output_path=MODEL_OUTPUT_PATH)
    trainer.run()



        
        
        

        