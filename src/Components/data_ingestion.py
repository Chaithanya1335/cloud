import sys,os
from src.Exception import CustomException
from src.loger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer
import pandas as pd


@dataclass
class DataIngestionConfig:
    train_csv:str = os.path.join("artifacts","train.csv")
    test_csv:str = os.path.join("artifacts","test.csv")
    raw_data:str = os.path.join("artifacts","raw.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig
    
    def initiate_data_ingestion(self):
        """
        Extracts The data from Local Csv file
        
        Returns : (train_data_path,test_data_path)
        """

        logging.info(" Data Ingestion started ! ")
        try:
            df = pd.read_csv(r"notebook\data\stud.csv")
            logging.info(" Reading Data completed  ")
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data,index=False,header=True)

            logging.info(" Data splitting started  ")
            train_data , test_data = train_test_split(df,test_size=0.2,random_state=20)
            logging.info(" Data splitting Completed  ")

            train_data.to_csv(self.data_ingestion_config.train_csv,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.test_csv,index=False,header=True)

            logging.info(" Train and Test Data saved  ")

            logging.info("Data Ingestion Completed ! ")

            return (self.data_ingestion_config.train_csv,self.data_ingestion_config.test_csv)
        
        except Exception as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion() 
    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessor_path = data_transformation.initiate_data_transformation(train_path=train_path,test_path=test_path)
    r2_score = ModelTrainer().initiate_model_training(train_arr,test_arr)