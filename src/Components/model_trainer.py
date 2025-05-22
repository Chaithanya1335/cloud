import os
import sys
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor,ExtraTreesRegressor)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.Exception import CustomException
from src.loger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts",'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()
    

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting the data")

            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            logging.info("Models initializing")

            models = {
                "Linear Regression":LinearRegression(),
                "LogisticRegression": LogisticRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "ExtraTreesRegressor":ExtraTreesRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor()
            }

            params = {
                "Linear Regression": {'fit_intercept':[True,False]},
                "LogisticRegression": {"penalty" : ['l1', 'l2', 'elasticnet', None]},
                "DecisionTreeRegressor":{"criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"]},
                "KNeighborsRegressor": {"n_neighbors":[5,10,15]},
                "GradientBoostingRegressor":{"loss" : ['squared_error', 'absolute_error', 'huber', 'quantile']},
                "AdaBoostRegressor":{"n_estimators":[50,100,150]},
                "RandomForestRegressor":{"n_estimators":[50,100,150]},
                "ExtraTreesRegressor":{"n_estimators":[50,100,150]},
                "XGBRegressor": {'learning_rate':[.1,.01,.05,.001]},
                "CatBoostRegressor":{'learning_rate':[.1,.01,.05,.001]}
            }


            evaluate_report:dict = evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models = models,params=params)

            best_score = max(evaluate_report.values())

            best_model_name = list(evaluate_report.keys())[list(evaluate_report.values()).index(best_score)]

            best_model = models[best_model_name]

            y_pred = best_model.predict(x_test)

            score = r2_score(y_test,y_pred)
            
            print(f"{best_model_name}:{score}")

            if score<0.6:
                raise CustomException("Score is less than 0.6 Try HyperParameterTuning ")
            save_object(
                self.model_config.model_path,
                best_model
            )

            logging.info("Model saved ")

            logging.info("Model Training Completed")

            return score

        except Exception as e:
            raise CustomException(e,sys)
        




            
