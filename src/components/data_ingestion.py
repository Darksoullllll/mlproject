import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationconfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass  # this decorator allows us to define variables without an __init__ function
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('data', "raw", "raw_data.csv")  # Specify the filename here

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")
        try:
            # Read the dataset
            df = pd.read_csv(r'D:\ML-Project\notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")

            # Ensure all necessary directories exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Train-test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Data Ingestion
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    # Data Transformation
    data_transformation_obj = DataTransformation()
    train_arr,test_arr,_ =data_transformation_obj.initiate_data_tranformation(train_data,test_data)
    
    # Training the model
    model_training_obj = ModelTrainer()
    model_training_obj.initiate_model_trainer(train_arr,test_arr)
