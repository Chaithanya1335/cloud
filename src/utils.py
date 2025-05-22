import pickle
import os
import sys
from src.Exception import CustomException
from src.loger import logging
from sklearn.metrics import r2_score


def save_object(file_path,obj):

    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,'wb')as f:
            pickle.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(x_train,y_train,x_test,y_test,models):

    try:
        logging.info("Model Training started")

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train,y_train)

            y_preds = model.predict(x_test)

            r2score = r2_score(y_test,y_preds)

            report[list(models.keys())[i]] = r2score

        logging.info("Model Training and Evaluation Completed")

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
