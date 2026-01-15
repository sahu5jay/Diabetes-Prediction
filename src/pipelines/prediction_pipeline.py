import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.joblib')
            model_path=os.path.join('artifacts','model.joblib')

            logging.info("collected the path of the pickel file")

            preprocessor=load_object(preprocessor_path)
            logging.info("preprocessor pickle file loaded")
            model=load_object(model_path)
            logging.info("model pickle file loaded")

            logging.info("Custom data scaling started")
            data_scaled=preprocessor.transform(features)
            
            logging.info("Custom data scaled")

            pred=model.predict(data_scaled)
            logging.info("returing the predicted price application")
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)