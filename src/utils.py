from src.logger import logging
from src.exception import CustomException
import os 
import sys
import pickle

def save_object(file_path, obj):

    try:
        logging.info("extracting the path to save the model..")
        dir_path = os.path.dirname(file_path)
        logging.info("creating the directory inside the artifacts")
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            logging.info("Dumping the model to artifacts folder..")
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("Exception occured while saving the model")
        raise CustomException(e,sys)

