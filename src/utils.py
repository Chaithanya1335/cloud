import pickle
import os
import sys
from src.Exception import CustomException


def save_object(file_path,obj):

    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,'wb')as f:
            pickle.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
