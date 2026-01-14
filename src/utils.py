from src.logger import logging
from src.exception import CustomException
import os 
import sys
import pickle

def save_object(self,file_path, obj):
    try:
        pass
    except Exception as e:
        logging.info("")
        raise CustomException(e,sys)

