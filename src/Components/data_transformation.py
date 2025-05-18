from src.Exception import CustomException
from src.loger import logging
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object
import pandas as pd
import numpy as np
import os
import sys


@dataclass
class DataTransformationConfig:
    preprocessing_obj_path = os.path.join("artifacts","preprocessor.pkl")



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_preprocessing_object(self,train:pd.DataFrame):


        try:

            

            numerical_cols = train.select_dtypes(exclude="object").columns.to_list()
            cat_cols = train.select_dtypes(include="object").columns.to_list()


            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical and Categorical Pipelines Created ")

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,cat_cols)
            ])

            logging.info("Preprocessor object Created")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path:str,test_path:str):
        
        try:
            logging.info("Data Transformation Started")

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            target_column = "math_score"
            logging.info(" train and test data read sucessfully ")

            input_train_df = train_data.drop(columns=[target_column],axis=1)
            target_train_df = train_data[target_column]

            input_test_df = test_data.drop(columns=[target_column],axis=1)
            target_test_df = test_data[target_column]

            logging.info("Loading Preprocessor object")

            preprocessor = self.get_data_preprocessing_object(input_train_df)

            train_input = preprocessor.fit_transform(input_train_df)
            test_input = preprocessor.transform(input_test_df)

            train_arr = np.c_[train_input,np.array(target_train_df)]

            test_arr = np.c_[test_input,np.array(target_test_df)]

            logging.info(" data transformation completed ")

            save_object(
                file_path = self.data_transformation_config.preprocessing_obj_path,
                obj = preprocessor
            )

            logging.info("preprocessor object saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
            