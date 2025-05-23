import pickle
import os
import sys
from src.Exception import CustomException
from src.loger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):

    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,'wb')as f:
            pickle.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(x_train,y_train,x_test,y_test,models,params):

    try:
        logging.info("Model Training started")

        report = {}

        logging.info("Hyper Parameter Tuning Started")

        for i in range(len(list(models))):
            model = list(models.values())[i]
            parm = list(params.values())[i]

            gs = GridSearchCV(model,parm,cv=3,n_jobs=-1)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_preds = model.predict(x_test)

            r2score = r2_score(y_test,y_preds)

            report[list(models.keys())[i]] = r2score

        logging.info("Model Training and Evaluation Completed")

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
